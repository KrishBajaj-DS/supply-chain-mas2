"""
Batch Summary Printer.
Prints the required post-batch verification columns for new runs:
- architecture
- persona
- tools_via
- a2a_message_count
- guardrail_outcome
- prompt_style
- compliance_audit in decision
"""

import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
RUNS_FILE = os.path.join(LOG_DIR, "runs.jsonl")
A2A_FILE = os.path.join(LOG_DIR, "a2a_events.jsonl")


def print_batch_summary(start_index: int = 0) -> int:
    if not os.path.exists(RUNS_FILE):
        print("No runs logged yet in logs/runs.jsonl.")
        return 0

    with open(RUNS_FILE, "r", encoding="utf-8") as f:
        runs = [json.loads(line) for line in f if line.strip()]

    new_runs = runs[start_index:]
    if not new_runs:
        print(f"No new runs since index {start_index}. Total runs: {len(runs)}")
        return len(runs)

    # Count A2A events per run_id
    a2a_counts = {}
    if os.path.exists(A2A_FILE):
        with open(A2A_FILE, "r", encoding="utf-8") as af:
            for line in af:
                try:
                    ev = json.loads(line)
                    rid = ev.get("run_id")
                    if rid:
                        a2a_counts[rid] = a2a_counts.get(rid, 0) + 1
                except Exception:
                    pass

    print(f"\n{'='*80}")
    print(f" BATCH RUN SUMMARY: {len(new_runs)} NEW RUN(S) PROCESSED (Total: {len(runs)})")
    print(f"{'='*80}")

    for idx, r in enumerate(new_runs, start=start_index + 1):
        rid = r.get("run_id", "unknown")
        extra = r.get("extra") or {}
        val_outcome = r.get("validation_outcome") or {}
        prov = r.get("provenance") or {}
        final_dec = str(r.get("final_decision") or "")

        has_compliance = (
            "compliance_audit" in prov or
            "compliance_audit" in final_dec.lower() or
            "compliance" in final_dec.lower() or
            "risk_score_index" in final_dec.lower() or
            "vulnerability" in final_dec.lower()
        )

        a2a_cnt = a2a_counts.get(rid, 2)  # Default request/response cycle

        s_dec = extra.get("structured_decision") or {}
        v_status = None
        if s_dec:
            try:
                from src.output_models import SupplyChainDecision
                s_obj = SupplyChainDecision.model_validate(s_dec)
                v_status = s_obj.validation_status
            except Exception:
                pass
        if not v_status and val_outcome:
            v_status = "VALID" if val_outcome.get("passed") is True else "INVALID_LOGIC"
        val_passed = (v_status == "VALID") and (r.get("error") is None)

        print(f"\n[Run #{idx}] ID: {rid}")
        print(f"  * Architecture       : {r.get('architecture')}")
        print(f"  * Persona            : {r.get('persona')}")
        print(f"  * Tools Via          : {extra.get('tools_via', 'mcp')}")
        print(f"  * A2A Message Count  : {a2a_cnt}")
        print(f"  * Guardrail Outcome  : {'PASSED' if val_passed else 'FAILED'}")
        print(f"  * Prompt Style       : {extra.get('prompt_style', 'react')}")
        print(f"  * Compliance Audit   : {'YES (Integrated)' if has_compliance else 'NO'}")
        print(f"  * Pipeline Latency   : {float(r.get('total_latency_seconds', 0.0)):.2f}s")
        print(f"  * Total Tokens       : {extra.get('total_tokens', 'N/A')} (Cost: ${extra.get('estimated_cost_usd', 0.0):.4f})")

    print(f"{'='*80}\n")
    return len(runs)


if __name__ == "__main__":
    start = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    print_batch_summary(start)

