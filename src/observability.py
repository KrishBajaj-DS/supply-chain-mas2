"""
Lightweight AI observability layer.

Rather than depending on an external SaaS (AgentOps/Langfuse both need
their own API keys/accounts), this implements a simple but real tracer:
every run is logged as a JSON record with a run_id, timestamp, which
architecture was used, which persona/scenario triggered it, per-agent
wall-clock latency, and the final decision text. Logs are appended to
logs/runs.jsonl - one line per run - so they're easy to load into
pandas for the analysis the course requires.

This satisfies the "AI Observability: per-run logs/traces, >=40 runs,
linked to personas" requirement without adding another paid/free-tier
dependency to manage.
"""

import json
import time
import uuid
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs")
LOG_FILE = os.path.join(LOG_DIR, "runs.jsonl")


_active_tracer = None


def get_active_tracer():
    """Retrieve the currently active RunTracer context, if any."""
    global _active_tracer
    return _active_tracer


def log_tool_event(agent: str, tool_name: str, arguments: dict, result: str, status: str = "ok"):
    """Global helper to record tool invocation metadata to the active tracer."""
    tracer = get_active_tracer()
    if tracer:
        tracer.log_tool_call(agent, tool_name, arguments, result, status)


def make_step_callback(run_id: str = None):
    """
    Step callback for CrewAI (fires per agent action/thought/step).
    Records intermediate agent outputs, execution order, and errors without crashing.
    """
    def _step_callback(step_output):
        try:
            tracer = get_active_tracer()
            if not tracer:
                return
            agent = getattr(step_output, "agent", None) or "Agent"
            tool = getattr(step_output, "tool", None)
            tool_input = getattr(step_output, "tool_input", None)
            log = getattr(step_output, "log", None) or str(step_output)
            tracer.log_step_event(
                agent=str(agent),
                tool=str(tool) if tool else None,
                tool_input=tool_input,
                log_text=str(log)[:300]
            )
            if tool:
                args = tool_input if isinstance(tool_input, dict) else {"input": str(tool_input)}
                already = any(
                    tc.get("tool_name") == str(tool) and abs(tc.get("timestamp", 0) - time.time()) < 2.0
                    for tc in tracer.tool_calls
                )
                if not already:
                    tracer.log_tool_call(
                        agent=str(agent),
                        tool_name=str(tool),
                        arguments=args,
                        result=str(log)[:200],
                        status="ok"
                    )
        except Exception:
            pass
    return _step_callback


class RunTracer:
    """Use as a context manager around one full crew run."""

    def __init__(self, architecture: str, persona: str, scenario: str,
                 write_to_disk: bool = True, log_file: str = None):
        self.run_id = str(uuid.uuid4())[:8]
        self.architecture = architecture
        self.persona = persona
        self.scenario = scenario
        self.write_to_disk = write_to_disk
        self.log_file = log_file or LOG_FILE
        self.agent_timings = []
        self.tool_calls = []
        self.step_events = []
        self.provenance = {}
        self.validation_outcome = None
        self.start_time = None
        self.end_time = None
        self.final_decision = None
        self.judge_score = None
        self.error = None
        self.extra = {}

    def __enter__(self):
        global _active_tracer
        self.start_time = time.time()
        _active_tracer = self
        try:
            from src.cost import reset_token_counter
            reset_token_counter()
        except Exception:
            pass
        return self

    def log_step_event(self, agent: str, tool: str = None, tool_input: Any = None, log_text: str = ""):
        """Record intermediate step callback event (thought/tool/result/order) without crashing."""
        try:
            self.step_events.append({
                "step_index": len(self.step_events) + 1,
                "agent": agent,
                "tool": tool,
                "tool_input": str(tool_input)[:100] if tool_input is not None else None,
                "log_preview": str(log_text)[:200],
                "timestamp": time.time(),
            })
        except Exception:
            pass

    def log_agent_step(self, agent_role: str, latency_seconds: float, judge_score=None, kind=None):
        """judge_score, if provided, is a {"score": int, "rationale": str}
        dict from judge_agent_output() - per-agent accuracy, distinct from
        the whole-run judge_score on the RunTracer itself."""
        entry = {
            "agent": agent_role,
            "latency_seconds": round(latency_seconds, 3),
        }
        if judge_score is not None:
            entry["judge_score"] = judge_score
        if kind is not None:
            entry["kind"] = kind  # e.g. 'finish_offset' for parallel runs
        self.agent_timings.append(entry)

    def log_tool_call(self, agent: str, tool_name: str, arguments: dict, result: str, status: str = "ok"):
        """Record tool invocation metadata for end-to-end provenance."""
        self.tool_calls.append({
            "agent": agent,
            "tool_name": tool_name,
            "arguments": arguments,
            "result_preview": str(result)[:150] + ("..." if len(str(result)) > 150 else ""),
            "status": status,
            "timestamp": time.time(),
        })

    def set_provenance(self, lineage_dict: dict):
        """Record lineage links connecting upstream tool outputs to the final decision."""
        self.provenance.update(lineage_dict)

    def set_validation_outcome(self, passed: bool, schema: str = "SupplyChainDecision", details: str = ""):
        """Record structured output validation result."""
        self.validation_outcome = {
            "passed": passed,
            "schema": schema,
            "details": details,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def set_extra(self, **kwargs):
        """Attach extra per-run fields (tokens, cost, structured decision,
        tool transport...). Stored under the record's 'extra' key."""
        self.extra.update(kwargs)

    def set_final_decision(self, text: str):
        self.final_decision = text

    def set_judge_score(self, score, rationale=""):
        self.judge_score = {"score": score, "rationale": rationale}

    def to_dict(self):
        end = self.end_time or time.time()
        start = self.start_time or end
        return {
            "run_id": self.run_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "architecture": self.architecture,
            "persona": self.persona,
            "scenario": self.scenario,
            "total_latency_seconds": round(end - start, 3),
            "agent_timings": self.agent_timings,
            "tool_calls": self.tool_calls,
            "step_events": self.step_events,
            "provenance": self.provenance,
            "validation_outcome": self.validation_outcome,
            "final_decision": self.final_decision,
            "judge_score": self.judge_score,
            "error": self.error,
            "extra": self.extra,
        }

    def __exit__(self, exc_type, exc_val, exc_tb):
        global _active_tracer
        self.end_time = time.time()
        if exc_type is not None:
            self.error = str(exc_val)
            err_lower = self.error.lower()
            if any(k in err_lower for k in ["tpd", "tokens per day", "daily_limit", "quota", "exceeded your current quota"]):
                self.extra.setdefault("error_category", "rate_limit_quota")
            elif any(k in err_lower for k in ["ratelimit", "rate_limit", "tpm", "tokens per minute", "too many requests"]):
                self.extra.setdefault("error_category", "rate_limit_tpm")
            else:
                self.extra.setdefault("error_category", "execution_error")

        record = self.to_dict()

        if self.write_to_disk and self.log_file:
            os.makedirs(os.path.dirname(os.path.abspath(self.log_file)), exist_ok=True)
            with open(self.log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record) + "\n")

        _active_tracer = None
        return False  # don't suppress exceptions


def load_all_runs():
    """Load every logged run as a list of dicts, for analysis/reporting."""
    if not os.path.exists(LOG_FILE):
        return []
    runs = []
    with open(LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    return runs
