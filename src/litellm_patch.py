"""
Workaround for a known CrewAI bug (github.com/crewAIInc/crewAI PR #7176,
issue #6789): CrewAI's prompt-caching feature tags messages with an
internal 'cache_breakpoint' key. Native providers (OpenAI/Anthropic) strip
it before sending, but providers routed through LiteLLM - which includes
Groq, since it isn't a CrewAI "native" provider - do NOT get it stripped,
so Groq's API rejects the request with:
  'property cache_breakpoint is unsupported'

Also adds automatic retry-with-backoff on Groq rate-limit errors (free
tier caps at a low tokens-per-minute limit, e.g. 8000 TPM), since a single
crew run can exceed that limit partway through and would otherwise crash.

Import this once, before building any Crew/Agent, e.g. at the top of
your entrypoint script:
    import src.litellm_patch  # noqa: F401  (import for side effect)
"""

import sys
import time
import re
import litellm

if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

_original_completion = litellm.completion

MAX_RETRIES = 5
DEFAULT_WAIT_SECONDS = 10

_token_call_history = []  # list of (timestamp, tokens)


def _throttle_sliding_window(estimated_tokens: int = 2400, tpm_limit: int = 7600):
    """Enforce rolling 60-second token window under Groq's 8,000 TPM limit."""
    now = time.time()
    while _token_call_history and now - _token_call_history[0][0] >= 60.0:
        _token_call_history.pop(0)
    current_tokens = sum(t[1] for t in _token_call_history)
    if current_tokens + estimated_tokens > tpm_limit and _token_call_history:
        oldest_ts = _token_call_history[0][0]
        wait_s = max(2.0, 60.0 - (now - oldest_ts) + 1.5)
        print(f"[throttle] Rolling 60s tokens at {current_tokens}. Pausing {wait_s:.1f}s to stay strictly under {tpm_limit} TPM...")
        time.sleep(wait_s)
        now = time.time()
        while _token_call_history and now - _token_call_history[0][0] >= 60.0:
            _token_call_history.pop(0)


def _strip_cache_breakpoint(messages):
    if not messages:
        return messages
    cleaned = []
    for m in messages:
        if isinstance(m, dict) and "cache_breakpoint" in m:
            m = {k: v for k, v in m.items() if k != "cache_breakpoint"}
        cleaned.append(m)
    return cleaned


def _extract_wait_seconds(error_message: str) -> float:
    """Groq's error message includes 'Please try again in X.Ys' or 'XmY.Ys'."""
    match = re.search(r"try again in (?:(\d+)m)?([\d.]+)s", str(error_message))
    if match:
        mins = float(match.group(1)) if match.group(1) else 0.0
        secs = float(match.group(2))
        return mins * 60.0 + secs + 2.0  # small buffer
    return DEFAULT_WAIT_SECONDS


def _patched_completion(*args, **kwargs):
    if "messages" in kwargs:
        kwargs["messages"] = _strip_cache_breakpoint(kwargs["messages"])
    elif len(args) > 0 and isinstance(args[0], list):
        args = (_strip_cache_breakpoint(args[0]),) + args[1:]

    last_error = None
    for attempt in range(MAX_RETRIES):
        try:
            _throttle_sliding_window()
            resp = _original_completion(*args, **kwargs)
            if hasattr(resp, "usage") and resp.usage:
                p = getattr(resp.usage, "prompt_tokens", 0) or 0
                c = getattr(resp.usage, "completion_tokens", 0) or 0
                _token_call_history.append((time.time(), p + c))
                try:
                    from src.cost import record_call_usage
                    record_call_usage(p, c)
                except Exception:
                    pass
            if kwargs.get("tools") and hasattr(resp, "choices") and resp.choices:
                msg = resp.choices[0].message
                if not getattr(msg, "tool_calls", None) and getattr(msg, "content", None):
                    c = str(msg.content).strip()
                    if c.startswith("{") and ("recommended_action" in c or "shortfall" in c):
                        pydantic_tool = None
                        for t in kwargs["tools"]:
                            tname = t.get("function", {}).get("name", "")
                            if "supplychaindecision" in tname.lower():
                                pydantic_tool = tname
                                break
                        if pydantic_tool:
                            from litellm.utils import ChatCompletionMessageToolCall, Function
                            msg.tool_calls = [
                                ChatCompletionMessageToolCall(
                                    id="call_pydantic",
                                    type="function",
                                    function=Function(name=pydantic_tool, arguments=c)
                                )
                            ]
                            resp.choices[0].finish_reason = "tool_calls"
            return resp
        except litellm.RateLimitError as e:
            last_error = e
            wait_s = _extract_wait_seconds(str(e))
            if wait_s > 30 or attempt >= 1:
                print(f"[rate limit] Rate limit wait ({wait_s:.1f}s) exceeded threshold - stopping immediately.")
                raise e
            print(f"[rate limit] Attempt {attempt + 1}/{MAX_RETRIES} - "
                  f"waiting {wait_s:.1f}s for TPM window cooldown before single retry...")
            time.sleep(wait_s)
        except Exception as e:
            err_str = str(e)
            if "attempted to call tool 'json'" in err_str or "tool_use_failed" in err_str:
                try:
                    import json
                    from litellm import ModelResponse, Choices, Message
                    from litellm.utils import ChatCompletionMessageToolCall, Function
                    # Groq puts the generated payload in 'failed_generation'
                    idx = err_str.find('"failed_generation":')
                    if idx != -1:
                        raw = err_str[idx + len('"failed_generation":'):].strip()
                        # Groq escapes string or passes json
                        if raw.startswith('"'):
                            raw = json.loads(raw[:raw.rfind('"')+1])
                        elif raw.startswith('{'):
                            raw = raw[:raw.rfind('}')+1]
                        parsed = json.loads(raw) if isinstance(raw, str) else raw

                        # Available tools in current call
                        available_tools = {}
                        for t in kwargs.get("tools") or []:
                            if isinstance(t, dict) and "function" in t:
                                fn_name = t["function"].get("name")
                                if fn_name:
                                    available_tools[fn_name.lower()] = fn_name

                        target_name = parsed.get("name") if isinstance(parsed, dict) else None
                        matched_tool = None
                        if target_name and str(target_name).lower() in available_tools:
                            matched_tool = available_tools[str(target_name).lower()]
                        elif "supplychaindecision" in available_tools and isinstance(parsed, dict) and ("recommended_action" in parsed or "shortfall" in parsed):
                            matched_tool = available_tools["supplychaindecision"]

                        # Extract arguments
                        if isinstance(parsed, dict) and "arguments" in parsed:
                            tool_args = parsed["arguments"]
                            args_str = json.dumps(tool_args) if isinstance(tool_args, (dict, list)) else str(tool_args)
                        else:
                            args_str = json.dumps(parsed) if isinstance(parsed, (dict, list)) else str(parsed)

                        resp = ModelResponse()
                        if matched_tool and kwargs.get("tools"):
                            tc = ChatCompletionMessageToolCall(
                                id="call_fallback",
                                type="function",
                                function=Function(name=matched_tool, arguments=args_str)
                            )
                            resp.choices = [Choices(finish_reason="tool_calls", index=0, message=Message(content=None, role="assistant", tool_calls=[tc]))]
                        else:
                            # Plain text completion - DO NOT return tool_calls!
                            if isinstance(parsed, dict) and "content" in parsed:
                                text_out = str(parsed["content"])
                            elif isinstance(parsed, dict) and "thought" in parsed:
                                text_out = str(parsed["thought"])
                            else:
                                text_out = args_str if isinstance(args_str, str) else str(parsed)
                            resp.choices = [Choices(finish_reason="stop", index=0, message=Message(content=text_out, role="assistant", tool_calls=None))]
                        return resp
                except Exception:
                    pass
            raise e
    raise last_error


litellm.completion = _patched_completion