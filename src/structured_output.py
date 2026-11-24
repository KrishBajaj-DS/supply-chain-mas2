"""
Structured final output (course criterion 3).

The Manager Agent writes a markdown decision report (Reason/Act/Observe
trace, Final Recommended Action, Justification table, Risks). This module
converts every such report into a validated Pydantic `DecisionReport`
(typed JSON), so decisions can be analyzed, compared across personas and
audited programmatically.

Honest scope: this is post-hoc extraction + Pydantic validation of the
Manager's markdown, not constrained generation (`output_pydantic`). The
parse-success rate is reported so the reliability of the extraction is
itself visible.

Usage:  python -m src.structured_output
Writes: logs/decisions_structured.jsonl, logs/decisions_by_persona.png
"""

import json
import re
from collections import Counter, defaultdict
from typing import Literal, Optional

from pydantic import BaseModel, Field, ValidationError, model_validator

from src.observability import load_all_runs


class DecisionReport(BaseModel):
    action_type: Literal["switch_supplier", "wait", "expedite", "hybrid"]
    headline: str = Field(min_length=10)
    supplier_id: Optional[str] = Field(default=None, pattern=r"^SUP-\d{3}$")
    lead_time_days: Optional[int] = Field(default=None, ge=0, le=90)
    total_cost_usd: Optional[float] = Field(default=None, ge=0)
    reliability_pct: Optional[float] = Field(default=None, ge=0, le=100)
    has_react_trace: bool
    has_justification: bool
    has_risks: bool

    @model_validator(mode="after")
    def _switch_needs_supplier(self):
        if self.action_type in ("switch_supplier", "expedite") and not self.supplier_id:
            raise ValueError(f"{self.action_type} decision has no supplier_id")
        return self


SUPPLIER_NAMES = {
    r"meridian": "SUP-101", r"northline": "SUP-102", r"vertex": "SUP-103",
    r"rapid source": "SUP-104", r"anchor bay": "SUP-105",
}


def _normalize(text: str) -> str:
    return (text.replace("\u2011", "-").replace("\u2010", "-").replace("\u2013", "-")
                .replace("\u202f", " ").replace("\u00a0", " ")
                .replace("\u201c", '"').replace("\u201d", '"').replace("\u2019", "'"))


def extract_decision_report(text: str) -> DecisionReport:
    """Raises ValueError / ValidationError if the text can't be structured."""
    t = _normalize(text)
    m = re.search(r"(?is)final recommended action(.*?)(?:\n#{2,4} |\n---\s*\n#{2,4} |\Z)", t)
    if not m:
        raise ValueError("no 'Final Recommended Action' section")
    section = m.group(1)

    # Headline = first substantial bold span (skip short labels like "Action:").
    headline = None
    for hm in re.finditer(r"\*\*(.+?)\*\*", section, re.S):
        cand = re.sub(r"\s+", " ", hm.group(1)).strip()
        if len(cand) >= 15 and not cand.endswith(":"):
            headline = cand
            break
    if not headline:
        raise ValueError("no substantial bold headline in recommended-action section")
    return _build_report(headline, section, t)


def _build_report(headline: str, section: str, full_text: str) -> DecisionReport:
    """Shared by the markdown path and the JSON path: classify the decision from
    its headline, then pull supplier / lead time / cost / reliability out of the
    section text."""
    t = full_text

    def _suppliers_in(txt: str):
        found = set(f"SUP-{n}" for n in re.findall(r"SUP-(\d{3})", txt))
        for name, sid in SUPPLIER_NAMES.items():
            if re.search(name, txt, re.I):
                found.add(sid)
        found.discard("SUP-201")  # incumbent (delayed) supplier, not an alternative
        return found

    in_headline = _suppliers_in(headline)
    h = headline.lower()
    wait_like = bool(re.search(
        r"\bwait\b|keep the (existing|current|order)|proceed with (the )?(existing|current|original|option \(?a)|"
        r"no change|stay with|do not (cancel|replace)|retain", h))

    if in_headline:
        supplier_id = sorted(in_headline)[0]
        if supplier_id == "SUP-104" or re.search(r"expedit", h):
            action = "expedite"
        elif wait_like:
            action = "hybrid"
        else:
            action = "switch_supplier"
    elif wait_like:
        # A bare "wait" is only trusted if the rest of the section proposes
        # no alternative supplier; otherwise it may really be a hybrid plan.
        if _suppliers_in(section[:1500]):
            raise ValueError("ambiguous: headline says wait but section names an alternative supplier")
        supplier_id, action = None, "wait"
    else:
        raise ValueError("ambiguous: headline names no supplier and does not say wait")

    lt = re.search(r"(?:lead[- ]?time|arriv\w*|deliver\w*)[^\d\n]{0,40}(\d{1,2})[ -]*(?:calendar )?days?", section, re.I)
    cost = (re.search(r"total[^$\n]{0,50}\$\s?([\d,]+(?:\.\d+)?)", section, re.I)
            or re.search(r"\$\s?([\d,]{5,}(?:\.\d+)?)", section))
    rel = re.search(r"(?:reliab\w*|on-time)[^\d\n]{0,40}(\d{2,3}(?:\.\d+)?)\s?%", section, re.I) \
        or re.search(r"(\d{2,3}(?:\.\d+)?)\s?%\s*(?:on-time|reliab)", section, re.I)

    rel_val = float(rel.group(1)) if rel else None
    if rel_val is not None and rel_val > 100:
        rel_val = None  # extraction artifact (not a percentage of reliability)

    return DecisionReport(
        action_type=action,
        headline=headline[:300],
        supplier_id=supplier_id,
        lead_time_days=int(lt.group(1)) if lt else None,
        total_cost_usd=float(cost.group(1).replace(",", "")) if cost else None,
        reliability_pct=rel_val,
        has_react_trace=bool(re.search(r"reason", t, re.I) and re.search(r"observe", t, re.I)),
        has_justification=bool(re.search(r"justification", t, re.I)),
        has_risks=bool(re.search(r"risks?", t, re.I)),
    )


def extract_from_json(text: str) -> DecisionReport:
    """Current Manager task (src/tasks.py) asks for SupplyChainDecision JSON.
    Validate it with that schema first, then reuse the same classification as
    the markdown path so both formats map to one DecisionReport."""
    from src.output_models import SupplyChainDecision
    t = _normalize(text)
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("no JSON object in decision")
    dec = SupplyChainDecision.model_validate_json(m.group(0))
    section = f"{dec.recommended_action}\n{dec.justification}"
    for o in dec.options_evaluated:
        if o.option_name and o.option_name.lower() in dec.recommended_action.lower() and o.lead_time_days is not None:
            section += f"\nlead time {o.lead_time_days} days"
            if o.total_cost is not None:
                section += f"\ntotal ${o.total_cost:.2f}"
            break
    rep = _build_report(re.sub(r"\s+", " ", dec.recommended_action).strip()[:300] if len(dec.recommended_action) >= 10
                        else (dec.recommended_action + " (decision)"), section, t)
    return rep.model_copy(update={
        "has_react_trace": bool(dec.react_trace),
        "has_justification": bool(dec.justification.strip()),
        "has_risks": bool(dec.risks_and_tradeoffs.strip()),
    })


def extract_any(text: str) -> DecisionReport:
    """Markdown report (older runs) or SupplyChainDecision JSON (current runs)."""
    if text.lstrip().startswith("{") or re.search(r'"recommended_action"', text):
        try:
            return extract_from_json(text)
        except (ValueError, ValidationError):
            pass  # fall through to the markdown extractor
    return extract_decision_report(text)


def _parse_json_object(raw: str) -> dict:
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    if not m:
        raise ValueError("no JSON object in LLM reply")
    return json.loads(m.group(0))


def _default_llm_call(prompt: str) -> str:
    from src.judge import judge_llm  # lazy: needs litellm + GROQ_API_KEY
    resp = judge_llm.call(messages=[{"role": "user", "content": prompt}])
    return resp if isinstance(resp, str) else str(resp)


def structure_decision(text: str, llm_call=None, max_llm_attempts: int = 2):
    """Turn a Manager decision report into a validated DecisionReport.

    Returns (DecisionReport | None, method) where method is 'regex' (free),
    'llm' (regex was ambiguous; one small LLM call produced the JSON, which
    was then validated by Pydantic) or 'failed'. The LLM output is never
    trusted: it must pass the same Pydantic validation as the regex path.
    """
    try:
        return extract_any(text), "regex"
    except (ValueError, ValidationError):
        pass

    call = llm_call or _default_llm_call
    schema = json.dumps(DecisionReport.model_json_schema(), indent=1)
    prompt = (
        "Convert this supply-chain decision report into ONE JSON object that "
        "follows this JSON schema exactly. Rules: action_type is one of "
        "switch_supplier, wait, expedite, hybrid (hybrid = keep the delayed "
        "shipment AND also order from an alternative supplier). supplier_id is "
        "the recommended ALTERNATIVE supplier (SUP-101..SUP-105), never SUP-201; "
        "null if none. Use null for any number not stated. Reply with JSON only.\n\n"
        f"SCHEMA:\n{schema}\n\nREPORT:\n{text[:6000]}"
    )
    for _ in range(max_llm_attempts):
        try:
            data = _parse_json_object(call(prompt))
            if data.get("supplier_id") == "SUP-201":
                data["supplier_id"] = None
            return DecisionReport.model_validate(data), "llm"
        except Exception:
            prompt += "\n\nYour previous reply was not valid. Return ONLY a valid JSON object."
    return None, "failed"


def main(out_jsonl="logs/decisions_structured.jsonl", out_png="logs/decisions_by_persona.png"):
    runs = [r for r in load_all_runs() if not r.get("error") and r.get("final_decision")]
    print(f"Successful runs with a final decision: {len(runs)}")

    parsed, failed = [], []
    for r in runs:
        try:
            rep = extract_any(r["final_decision"])
            parsed.append((r, rep))
        except (ValueError, ValidationError) as e:
            failed.append((r["run_id"], str(e).splitlines()[0][:100]))

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for r, rep in parsed:
            f.write(json.dumps({"run_id": r["run_id"], "architecture": r["architecture"],
                                "persona": r["persona"], **rep.model_dump()}) + "\n")

    pct = 100 * len(parsed) / max(1, len(runs))
    print(f"Parsed into validated DecisionReport: {len(parsed)}/{len(runs)} ({pct:.0f}%)")
    for rid, why in failed[:10]:
        print(f"  could not structure {rid}: {why}")
    print(f"Structured decisions written to {out_jsonl}")

    if parsed:
        print("Field coverage across parsed reports: " + ", ".join(
            f"{k}={100*sum(getattr(rep,k) is not None for _, rep in parsed)/len(parsed):.0f}%"
            for k in ("supplier_id", "lead_time_days", "total_cost_usd", "reliability_pct")))

    seq = [(r, rep) for r, rep in parsed if r["architecture"] == "sequential"]
    print(f"\n=== Decision chosen per persona (sequential, n={len(seq)}) ===")
    table = defaultdict(Counter)
    for r, rep in seq:
        label = (rep.supplier_id or "incumbent (wait)") if rep.action_type != "wait" else "wait for SUP-201"
        if rep.action_type == "hybrid":
            label += " +contingency"
        table[r["persona"].split(" - ")[0]][label] += 1
    labels = sorted({l for c in table.values() for l in c})
    print(f"{'persona':28}" + "".join(f"{l[:22]:>24}" for l in labels))
    for p, c in table.items():
        print(f"{p:28}" + "".join(f"{c.get(l, 0):>24}" for l in labels))

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        personas = list(table)
        fig, ax = plt.subplots(figsize=(8, 4.6))
        bottom = [0] * len(personas)
        palette = plt.cm.tab10.colors
        for i, l in enumerate(labels):
            vals = [table[p].get(l, 0) for p in personas]
            ax.bar(personas, vals, bottom=bottom, label=l, color=palette[i % 10])
            bottom = [b + v for b, v in zip(bottom, vals)]
        ax.set_ylabel("number of runs")
        ax.set_title("Final decision by persona (from structured DecisionReport)")
        ax.legend(fontsize=8, loc="upper right")
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"Chart written: {out_png}")
    except Exception as e:
        print(f"(chart skipped: {e})")


if __name__ == "__main__":
    main()
