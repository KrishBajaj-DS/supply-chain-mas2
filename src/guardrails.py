"""
Guardrails for the Supply Chain Disruption Management MAS (course criterion 8).

Two layers, both pure Python (zero LLM cost, never touch the Groq quota, and
cannot themselves be prompt-injected):

1. INPUT guardrail  - validate_user_input() / enforce_guardrails(): screens the
   scenario/persona text BEFORE any agent is created or any LLM call is made.
   Blocks prompt injection, abusive/toxic language, bad type/length and
   off-domain requests. Operational urgency ("URGENT:", "production lines are
   at risk") is deliberately allowed - only real abuse is blocked.

2. TOOL-OUTPUT guardrail - validate_tool_output() / @guarded(...): parses each
   tool's return value into a Pydantic model (type + range + consistency
   checks). A malformed or implausible tool result is rejected and the agent
   is told to retry instead of reasoning from bad data.

Every blocked item (and every screened input) is appended to
logs/guardrail_events.jsonl for audit (links to criterion 9, observability).

Backward-compatible API kept from the earlier version of this module:
validate_scenario_input(), enforce_guardrails(), load_guardrail_events(),
guardrail_summary().
"""

import functools
import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Tuple

from pydantic import BaseModel, Field, ValidationError, model_validator

# Absolute path so the log lands in <project>/logs no matter where the
# script is launched from (a relative path silently wrote elsewhere).
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
GUARDRAIL_LOG = os.path.join(LOG_DIR, "guardrail_events.jsonl")

MIN_INPUT_CHARS = 10
MAX_INPUT_CHARS = 2000


class GuardrailViolation(ValueError):
    """Raised when user input is blocked before reaching any agent."""

    def __init__(self, message: str, violation_type: str = "validation_failed"):
        self.violation_type = violation_type
        super().__init__(message)


@dataclass
class GuardrailResult:
    allowed: bool
    category: str = "ok"
    reason: str = ""
    outcome: str = "passed"  # "passed" | "sanitised" | "blocked"
    sanitized_text: str = ""
    run_id: str = ""


# --------------------------------------------------------------------------
# Audit log
# --------------------------------------------------------------------------
def _log_event(kind: str, category: str, reason: str, snippet: str = "", passed: bool = False, outcome: str = "passed", run_id: str = ""):
    """Append one audit record. Logging must never break a run."""
    try:
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(GUARDRAIL_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps({
                "event_id": str(uuid.uuid4())[:8],
                "run_id": run_id,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "kind": kind,                       # "input" | "tool_output"
                "passed": passed,
                "outcome": outcome,                 # "passed" | "sanitised" | "blocked"
                "violation_type": category,
                "detail": reason,
                "input_preview": snippet[:200] + ("..." if len(snippet) > 200 else ""),
            }) + "\n")
    except Exception:
        pass


# --------------------------------------------------------------------------
# 1. INPUT GUARDRAIL
# --------------------------------------------------------------------------
INJECTION_PATTERNS = [
    r"ignore (all |any |the |your )?(previous|prior|above|earlier) (instructions?|prompts?|rules?)",
    r"disregard (all |any |the |your )?(previous|prior|above|earlier|instructions?|rules?)",
    r"forget (everything|all|your) (instructions?|rules?|above)",
    r"forget (everything|all) (you|above)",
    r"(reveal|show|print|display|leak|output).{0,40}(system prompt|api[ _-]?key|secret|password|\.env|credentials?)",
    r"\byou are now\b",
    r"\bact as (an? )?(unrestricted|jailbroken|dan|root|admin)",
    r"\bact as if you (have no|are not)\b",
    r"\bpretend (that )?you (are|were|have)\b",
    r"\bjailbreak\b",
    r"\bdan mode\b",
    r"\b(developer|debug|god) mode\b",
    r"override (the |your )?(system )?(prompt|safety|guardrails?|rules?|policy|policies)",
    r"(bypass|disable|turn off) (the |your )?(safety|guardrails?|filters?|validation)",
    r"###\s*new instructions",
    r"</?\s*(system|assistant|instructions?)\s*>",
    r"\[\s*system\s*\]",
]

# Deliberately mild word list (abuse + light profanity). Extend as needed.
TOXIC_PATTERNS = [
    r"\b(idiots?|stupid|morons?|imbeciles?|dumbass(es)?)\b",
    r"\bshut up\b",
    r"\bpiece of garbage\b",
    r"\b(damn|crap|bullshit|wtf)\b",
    r"\bf[u*]+ck\w*",
    r"\bs[h*]+it\b",
    r"\bkill (all|every)\b",
    r"\bhate (all|every)\b",
    r"\bgo to hell\b",
]

DOMAIN_TERMS = [
    "supplier", "shipment", "delay", "widget", "inventory", "order",
    "stock", "demand", "sourcing", "expedit", "port", "units", "cost",
]

_INJECTION_RE = [re.compile(p, re.I) for p in INJECTION_PATTERNS]
_TOXIC_RE = [re.compile(p, re.I) for p in TOXIC_PATTERNS]


def validate_user_input(text: str, log: bool = True, run_id: str = None, allow_sanitisation: bool = False) -> GuardrailResult:
    """
    Screen scenario text before any agent sees it.
    - If allow_sanitisation=True and input has domain terms: sanitise toxic terms and continue.
    - Otherwise, toxic/abusive input is blocked (category='toxic_language').
    - Prompt injections are strictly blocked in all cases.
    - Clean request -> passed
    """
    if not isinstance(text, str):
        res = GuardrailResult(False, "type_error", f"Input is not a string (got {type(text).__name__}).", outcome="blocked", run_id=run_id or "")
    elif not text.strip():
        res = GuardrailResult(False, "empty_input", "Input is empty.", outcome="blocked", run_id=run_id or "")
    elif len(text.strip()) < MIN_INPUT_CHARS:
        res = GuardrailResult(False, "too_short", f"Input too short ({len(text.strip())} chars, minimum {MIN_INPUT_CHARS}).", outcome="blocked", run_id=run_id or "")
    elif len(text) > MAX_INPUT_CHARS:
        res = GuardrailResult(False, "too_long", f"Input too long ({len(text)} chars, maximum {MAX_INPUT_CHARS}).", outcome="blocked", run_id=run_id or "")
    else:
        norm = re.sub(r"\s+", " ", text)
        is_injection = False
        for pat in _INJECTION_RE:
            m = pat.search(norm)
            if m:
                is_injection = True
                res = GuardrailResult(False, "prompt_injection", f"Prompt injection detected: matched '{m.group()}'", outcome="blocked", run_id=run_id or "")
                break

        if not is_injection:
            has_domain = any(t in norm.lower() for t in DOMAIN_TERMS)
            toxic_matches = []
            for pat in _TOXIC_RE:
                m = pat.search(norm)
                if m:
                    toxic_matches.append(m.group())

            if toxic_matches:
                if has_domain and allow_sanitisation:
                    # Mixed message: legitimate request + toxic language -> SANITISE AND CONTINUE
                    cleaned = norm
                    for pat in _TOXIC_RE:
                        cleaned = pat.sub("[REDACTED]", cleaned)
                    res = GuardrailResult(
                        allowed=True,
                        category="toxic_language_sanitised",
                        reason=f"Sanitised toxic language: {', '.join(toxic_matches)}",
                        outcome="sanitised",
                        sanitized_text=cleaned,
                        run_id=run_id or "",
                    )
                else:
                    # Toxic/abusive language blocked
                    res = GuardrailResult(
                        allowed=False,
                        category="toxic_language",
                        reason=f"Toxic/abusive language detected: {', '.join(toxic_matches)}",
                        outcome="blocked",
                        run_id=run_id or "",
                    )
            elif not has_domain:
                res = GuardrailResult(
                    allowed=False,
                    category="off_domain",
                    reason="Request is not about a supply-chain disruption.",
                    outcome="blocked",
                    run_id=run_id or "",
                )
            else:
                res = GuardrailResult(
                    allowed=True,
                    category="ok",
                    reason="All checks passed",
                    outcome="passed",
                    sanitized_text=norm,
                    run_id=run_id or "",
                )

    if log:
        snippet = text if isinstance(text, str) else str(text)
        _log_event(
            kind="input",
            category=res.category,
            reason=res.reason,
            snippet=snippet,
            passed=res.allowed,
            outcome=res.outcome,
            run_id=run_id or "",
        )
    return res


def validate_scenario_input(text: str, run_id: str = None) -> Tuple[bool, str]:
    """Backward-compatible wrapper: returns (ok, reason)."""
    res = validate_user_input(text, run_id=run_id)
    return res.allowed, ("OK" if res.allowed else res.reason)


def enforce_guardrails(text: str, run_id: str = None, allow_sanitisation: bool = False) -> str:
    """Validate and RAISE GuardrailViolation if blocked; return sanitized/clean text if allowed."""
    res = validate_user_input(text, run_id=run_id, allow_sanitisation=allow_sanitisation)
    if not res.allowed:
        raise GuardrailViolation(f"[GUARDRAIL BLOCKED] {res.category}: {res.reason}",
                                 violation_type=res.category)
    return res.sanitized_text or text


# --------------------------------------------------------------------------
# 2. TOOL-OUTPUT GUARDRAIL (Pydantic validation of each tool's return value)
# --------------------------------------------------------------------------
class ShipmentStatus(BaseModel):
    supplier: str
    product: str
    ordered_units: int = Field(gt=0, le=10_000_000)
    delay_days: int = Field(ge=0, le=365)
    status: str


class DemandForecast(BaseModel):
    days: int = Field(gt=0, le=365)
    total_units: int = Field(ge=0, le=10_000_000)
    avg_per_day: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self):
        # total should equal avg/day * days (avg is rounded to 1 decimal)
        expected = self.avg_per_day * self.days
        if abs(self.total_units - expected) > max(5, 0.02 * expected):
            raise ValueError(f"total {self.total_units} != avg*days {expected:.0f}")
        return self


class InventoryCheck(BaseModel):
    shortage: bool
    current_stock: int = Field(ge=0, le=10_000_000)
    figure_units: int = Field(ge=0, le=10_000_000)  # shortfall (>=0) or ending stock (>=0)


class SupplierHits(BaseModel):
    n_hits: int = Field(ge=1)
    relevances: list[float]

    @model_validator(mode="after")
    def _range(self):
        if any(r < -1.0 or r > 1.0 for r in self.relevances):
            raise ValueError("relevance outside [-1, 1]")
        return self


class CostEstimate(BaseModel):
    units: int = Field(gt=0, le=10_000_000)
    unit_cost: float = Field(ge=0)
    baseline_unit_cost: float = Field(gt=0)
    total_cost: float = Field(ge=0)
    baseline_total: float = Field(ge=0)

    @model_validator(mode="after")
    def _consistent(self):
        expected = self.unit_cost * self.units
        if abs(self.total_cost - expected) > 0.01 * expected + 1:
            raise ValueError(f"total {self.total_cost} != unit*units {expected:.0f}")
        return self


def _m(pattern: str, text: str, flags=0):
    m = re.search(pattern, text, flags | re.S)
    if not m:
        raise ValueError(f"expected pattern not found: {pattern[:40]}...")
    return m


_NUM = r"(-?\d+(?:\.\d+)?)"


def _parse(tool_name: str, text: str):
    if tool_name == "check_shipment_status":
        m = _m(r"Supplier: (.+?) \| Product: (.+?) \| Ordered: (-?\d+) units \| "
               r"Delay: (-?\d+) days .*?Status: (\w+)", text)
        return ShipmentStatus(supplier=m[1], product=m[2], ordered_units=int(m[3]),
                              delay_days=int(m[4]), status=m[5])
    if tool_name == "forecast_demand":
        m = _m(r"next (\d+) days: ~(-?\d+) units total \(avg " + _NUM + r"/day\)", text)
        return DemandForecast(days=int(m[1]), total_units=int(m[2]), avg_per_day=float(m[3]))
    if tool_name == "check_inventory":
        stock = int(_m(r"current stock (-?\d+) units", text)[1])
        if "SHORTAGE PROJECTED" in text:
            fig = int(_m(r"shortfall: (-?\d+) units", text)[1])
            return InventoryCheck(shortage=True, current_stock=stock, figure_units=fig)
        fig = int(_m(r"ending stock: (-?\d+) units", text)[1])
        return InventoryCheck(shortage=False, current_stock=stock, figure_units=fig)
    if tool_name == "search_alternative_suppliers":
        rel = [float(x) for x in re.findall(r"\[SUP-\d+\][^\n]*?\(relevance=(-?[\d.]+)\)", text)]
        return SupplierHits(n_hits=len(rel), relevances=rel)
    if tool_name == "calculate_cost_impact":
        m = _m(r"on (-?\d+) units of .*?unit cost " + _NUM + r" \(baseline " + _NUM + r"\), "
               r"total cost " + _NUM + r" vs baseline total " + _NUM, text)
        return CostEstimate(units=int(m[1]), unit_cost=float(m[2]),
                            baseline_unit_cost=float(m[3]), total_cost=float(m[4]),
                            baseline_total=float(m[5]))
    return None  # unknown tool: nothing to validate


def validate_tool_output(tool_name: str, output, log: bool = True) -> GuardrailResult:
    """Parse a tool's return value into its Pydantic model."""
    if not isinstance(output, str) or not output.strip():
        res = GuardrailResult(False, "tool_type_error",
                              f"{tool_name} returned empty/non-string output.")
    elif re.match(r"\s*No .{0,60}(found|record)", output):
        # e.g. "No shipment record found for ..." / "No matching suppliers found."
        return GuardrailResult(True, "not_found", "Tool reported no record (valid).")
    else:
        try:
            _parse(tool_name, output)
            return GuardrailResult(True)
        except ValidationError as e:
            err = e.errors()[0]
            where = ".".join(str(x) for x in err["loc"]) or "value"
            res = GuardrailResult(False, "tool_output_invalid",
                                  f"{tool_name}: {where}: {err['msg']}"[:170])
        except ValueError as e:
            res = GuardrailResult(False, "tool_output_invalid",
                                  f"{tool_name}: {str(e).splitlines()[0][:160]}")
    if log:
        _log_event("tool_output", res.category, res.reason, str(output))
    return res


def guarded(tool_name: str):
    """Decorator for tool functions (place UNDER @tool). Validates the return
    value; on failure the agent receives a rejection message instead of the
    bad data. A crash inside the validator never breaks the tool."""
    def deco(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            out = fn(*args, **kwargs)
            try:
                from src.observability import log_tool_event
                arguments = kwargs if kwargs else {"args": [str(a) for a in args]}
                log_tool_event(agent=tool_name, tool_name=tool_name, arguments=arguments, result=str(out), status="ok")
            except Exception:
                pass
            try:
                res = validate_tool_output(tool_name, out)
            except Exception:
                return out
            if res.allowed:
                return out
            return (f"[GUARDRAIL REJECTED TOOL OUTPUT] {res.reason} "
                    f"Do not use these figures. Retry the tool call, or state "
                    f"that this data is unavailable.")
        return wrapper
    return deco


# --------------------------------------------------------------------------
# Audit-log helpers
# --------------------------------------------------------------------------
def load_guardrail_events():
    """Load all guardrail audit events as a list of dicts."""
    if not os.path.exists(GUARDRAIL_LOG):
        return []
    events = []
    with open(GUARDRAIL_LOG, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def guardrail_summary() -> str:
    """One-line summary of the guardrail audit log."""
    events = load_guardrail_events()
    if not events:
        return "No guardrail events logged yet."
    passed = sum(1 for e in events if e.get("passed"))
    blocked = len(events) - passed
    kinds = {}
    for e in events:
        if not e.get("passed"):
            vt = e.get("violation_type") or e.get("category") or "unknown"
            kinds[vt] = kinds.get(vt, 0) + 1
    kind_str = ", ".join(f"{k}={v}" for k, v in kinds.items()) or "none"
    return (f"Guardrail audit: {len(events)} total events | "
            f"{passed} passed | {blocked} blocked ({kind_str})")
