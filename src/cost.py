"""
Token and cost accounting (course criterion 10: cost and running-time estimates).

Runs use Groq's free tier, so the money actually spent is $0. The figure
reported here is the ESTIMATED cost of the same token usage at Groq's paid
list price for gpt-oss-120b, which is what a production deployment would pay.

List price (USD per 1M tokens, Groq, checked Oct 2026): $0.15 input, $0.60
output. Update the two constants if the price changes (console.groq.com).
Only tokens reported by CrewAI's crew usage metrics are counted (agent LLM
calls). The separate LLM-as-Judge calls are not included, so real totals
are slightly higher.
"""

PRICE_PER_M_INPUT = 0.15
PRICE_PER_M_OUTPUT = 0.60

_baseline_usage = None
_current_run_usage = {
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "total_tokens": 0,
    "llm_requests": 0,
}


def estimate_cost_usd(prompt_tokens: int, completion_tokens: int) -> float:
    return (prompt_tokens * PRICE_PER_M_INPUT + completion_tokens * PRICE_PER_M_OUTPUT) / 1_000_000


def record_call_usage(prompt_tokens: int, completion_tokens: int):
    """Accumulates exact per-call token usage returned from each Groq/LLM completion."""
    global _current_run_usage
    p = int(prompt_tokens or 0)
    c = int(completion_tokens or 0)
    _current_run_usage["prompt_tokens"] += p
    _current_run_usage["completion_tokens"] += c
    _current_run_usage["total_tokens"] += (p + c)
    _current_run_usage["llm_requests"] += 1


def reset_token_counter(initial_usage: dict = None):
    """
    Reset the baseline and run-scoped token counters.
    Ensures tokens and cost are reset at the start of every run.
    """
    global _baseline_usage, _current_run_usage
    _current_run_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "llm_requests": 0,
    }
    _baseline_usage = dict(initial_usage) if initial_usage else None


def usage_from_result(result=None, reset_baseline: bool = True):
    """
    Extract run-scoped token usage.
    If live Groq call usage was tracked via record_call_usage, returns the exact
    sum of calls for this run (fixing CrewAI's 6x agent multiplication).
    Otherwise computes delta from result.token_usage for mock/offline runs.
    """
    global _baseline_usage, _current_run_usage
    if _current_run_usage["llm_requests"] > 0:
        p = _current_run_usage["prompt_tokens"]
        c = _current_run_usage["completion_tokens"]
        t = _current_run_usage["total_tokens"]
        n = _current_run_usage["llm_requests"]
        return {
            "prompt_tokens": p,
            "completion_tokens": c,
            "total_tokens": t,
            "llm_requests": n,
            "estimated_cost_usd": round(estimate_cost_usd(p, c), 6),
        }

    u = getattr(result, "token_usage", None) if result else None
    if u is None:
        return None
    try:
        raw_p = int(getattr(u, "prompt_tokens", 0) or 0)
        raw_c = int(getattr(u, "completion_tokens", 0) or 0)
        raw_t = int(getattr(u, "total_tokens", 0) or (raw_p + raw_c))
        raw_n = int(getattr(u, "successful_requests", 0) or 0)
    except (TypeError, ValueError):
        return None

    if _baseline_usage is None:
        p, c, t, n = raw_p, raw_c, raw_t, raw_n
    else:
        p = max(0, raw_p - _baseline_usage.get("prompt_tokens", 0))
        c = max(0, raw_c - _baseline_usage.get("completion_tokens", 0))
        t = max(0, raw_t - _baseline_usage.get("total_tokens", 0))
        n = max(0, raw_n - _baseline_usage.get("llm_requests", 0))

    if reset_baseline:
        _baseline_usage = {
            "prompt_tokens": raw_p,
            "completion_tokens": raw_c,
            "total_tokens": raw_t,
            "llm_requests": raw_n,
        }

    return {
        "prompt_tokens": p,
        "completion_tokens": c,
        "total_tokens": t,
        "llm_requests": n,
        "estimated_cost_usd": round(estimate_cost_usd(p, c), 6),
    }
