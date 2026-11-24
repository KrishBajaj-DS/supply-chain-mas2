"""
Prompt-variant experiment (course criterion 10: variance in prompts).

Compares three prompt strategies for the decision-making Manager agent, across
the three personas:
  react  - explicit Reason -> Act -> Observe trace (the project's strategy)
  cot    - chain-of-thought: think step by step, no R/A/O labels
  direct - no reasoning trace, answer only

Cost-saving design: the five specialist agents' work is deterministic tool
output, so it is computed ONCE with no LLM calls (frozen team report) and only
the Manager step is run per style/persona. Each run is one small LLM call, so
the experiment costs a fraction of a full pipeline run. Results are logged to
logs/prompt_ablation.jsonl (separate from runs.jsonl so the main dataset is
untouched).

    python -m src.prompt_ablation --repeats 2        # 3 styles x 3 personas x 2 = 18 runs
    python -m src.prompt_ablation --analyze          # table + boxplots, no LLM
"""

import argparse
import json
import os
import re
import time
from datetime import datetime, timezone

LOG = "logs/prompt_ablation.jsonl"

STYLES = {
    "react": dict(
        goal="Synthesize all agent findings using explicit Reason -> Act -> Observe steps and propose one justified action.",
        backstory="You are a supply chain operations manager who reasons in explicit Reason, Act and Observe steps before deciding.",
        style="react",
    ),
    "cot": dict(
        goal="Synthesize all agent findings using step-by-step Chain of Thought reasoning and propose one justified action.",
        backstory="You are a supply chain operations manager who thinks through problems step by step before reaching a conclusion.",
        style="cot",
    ),
    "direct": dict(
        goal="Synthesize all agent findings directly and decisively, proposing one justified action.",
        backstory="You are a supply chain operations manager who delivers direct, bottom-line executive decisions with clear quantitative justification.",
        style="direct",
    ),
}


def build_team_report() -> str:
    """Deterministic stand-in for the five specialist agents (no LLM calls)."""
    os.environ.setdefault("GEMINI_API_KEY", "unused")
    from src import tools as t
    ship = t.check_shipment_status.func("SUP-201")
    fc = t.forecast_demand.func("Widget-A", 10)
    units = 2912
    inv = t.check_inventory.func("Widget-A", units, 10)
    try:
        sup = t.search_alternative_suppliers.func(query="alternative suppliers for Widget-A", top_k=5)
    except Exception as e:
        sup = f"(supplier search unavailable: {str(e)[:80]})"
    options = [
        ("wait for delayed shipment", 0),
        ("switch to Anchor Bay Manufacturing", 0),
        ("switch to Meridian Components Ltd.", 0),
        ("switch to Vertex Global Traders", -12),
        ("expedite via Rapid Source Logistics", 25),
    ]
    cost = "\n".join(t.calculate_cost_impact.func("Widget-A", 5000, o, p) for o, p in options)
    return (
        f"SHIPMENT REPORT:\n{ship}\n\n"
        f"DEMAND FORECAST:\n{fc}\n\n"
        f"INVENTORY:\n{inv}\n\n"
        f"ALTERNATIVE SUPPLIERS:\n{sup}\n\n"
        f"COST OPTIONS:\n{cost}\n\n"
        f"COMPLIANCE AUDIT:\nSupplier SUP-201 Risk Score: 35/100 (MODERATE) | ISO 9001: Certified | Recommendation: Maintain buffer monitoring."
    )


def _quota_exhausted(e) -> bool:
    return "tokens per day" in str(e) or "TPD" in str(e) or "rate limit" in str(e).lower()


def run(repeats: int, styles):
    import src.litellm_patch  # noqa: F401
    from crewai import Agent, Crew, Process, Task
    from src import agents as A
    from src.tasks import get_manager_prompt
    from src.scenarios import get_all_scenarios
    from src.judge import judge_decision
    from src.cost import usage_from_result, reset_token_counter
    from src.output_models import SupplyChainDecision

    team = build_team_report()
    os.makedirs("logs", exist_ok=True)
    # Use standard professional and aggressive personas (skip injection attempts for style ablation)
    scenarios = [s for s in get_all_scenarios() if not s["id"].startswith("injection_")]
    n = len(styles) * len(scenarios) * repeats
    print(f"Prompt ablation: {len(styles)} styles x {len(scenarios)} scenarios x {repeats} = {n} Manager-only runs")
    done = 0
    for _ in range(repeats):
        for style in styles:
            for sc in scenarios:
                st = STYLES[style]
                rec = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "style": style,
                    "persona": sc["persona"],
                    "scenario": sc["id"],
                    "error": None,
                }
                reset_token_counter()
                try:
                    agent = Agent(
                        role="Manager Agent",
                        goal=st["goal"],
                        backstory=st["backstory"],
                        llm=A.llm,
                        allow_delegation=False,
                        max_iter=3,
                        verbose=False,
                    )
                    prompt_desc = (
                        f"SCENARIO:\n{sc['text']}\n\n"
                        f"TEAM REPORTS (already verified from Blackboard):\n{team}\n\n"
                        f"{get_manager_prompt(prompt_style=style, persona=sc['persona'])}"
                    )
                    task = Task(
                        description=prompt_desc,
                        expected_output="A valid SupplyChainDecision structured object.",
                        output_pydantic=SupplyChainDecision,
                        agent=agent,
                    )
                    t0 = time.time()
                    res = Crew(agents=[agent], tasks=[task], process=Process.sequential).kickoff()
                    rec["latency_seconds"] = round(time.time() - t0, 2)
                    text = str(res)
                    rec["decision_chars"] = len(text)
                    usage = usage_from_result(res)
                    if usage:
                        rec.update(usage)

                    validated = getattr(res, "pydantic", None)
                    rec["structured_ok"] = isinstance(validated, SupplyChainDecision) and validated.validation_status in ["passed", "VALID"]
                    val_stat = getattr(validated, "validation_status", "VALID") if validated else "INVALID_LOGIC"
                    j = judge_decision(sc["text"], text, validation_status=val_stat)
                    rec["judge_score"] = j.get("score") if j else None
                except Exception as e:
                    rec["error"] = str(e)[:300]

                with open(LOG, "a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                done += 1
                print(f"[{done}/{n}] {style:6} / {sc['id']:20} "
                      + (f"ERR {rec['error'][:60]}" if rec["error"] else
                         f"latency={rec['latency_seconds']}s judge={rec['judge_score']} "
                         f"tokens={rec.get('total_tokens', '?')}"))
                if rec["error"] and _quota_exhausted(rec["error"]):
                    print("[QUOTA] API rate limit reached - stopping. Results so far are saved.")
                    return
                time.sleep(3)


def analyze(out_png="logs/prompt_ablation_boxplot.png"):
    import pandas as pd
    if not os.path.exists(LOG):
        print("No prompt-ablation data yet. Run: python -m src.prompt_ablation --repeats 2")
        return
    df = pd.DataFrame([json.loads(l) for l in open(LOG, encoding="utf-8")])
    if df.empty:
        print("logs/prompt_ablation.jsonl is empty.")
        return
    ok = df[df["error"].isna()].copy()
    print(f"\n=== Prompt strategy comparison (Manager step, n={len(ok)} successful of {len(df)}) ===")
    if ok.empty:
        return
    cols = {
        "n": ("style", "count"),
        "latency_s": ("latency_seconds", "mean"),
        "judge": ("judge_score", "mean"),
        "judge_sd": ("judge_score", "std"),
        "structured_pct": ("structured_ok", lambda s: round(100 * s.mean(), 1)),
    }
    if "total_tokens" in ok:
        cols["tokens"] = ("total_tokens", "mean")
    if "estimated_cost_usd" in ok:
        cols["est_cost_usd"] = ("estimated_cost_usd", "mean")
    print(ok.groupby("style").agg(**cols).round(3))

    print("\n=== By style x persona (mean judge / mean latency) ===")
    ok["persona_short"] = ok["persona"].str.split(" - ").str[0]
    print(ok.pivot_table(index="persona_short", columns="style",
                         values=["judge_score", "latency_seconds"], aggfunc="mean").round(2))
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        order = [s for s in STYLES if s in set(ok["style"])]
        metrics = [("judge_score", "Judge score /10"), ("latency_seconds", "Latency (s)")]
        if "total_tokens" in ok:
            metrics.append(("total_tokens", "Tokens per run"))
        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 4.4))
        if len(metrics) == 1:
            axes = [axes]
        for ax, (col, title) in zip(axes, metrics):
            vals = [ok.loc[ok["style"] == s, col].dropna().values for s in order]
            if any(len(v) > 0 for v in vals):
                ax.boxplot(vals, showmeans=True)
                ax.set_xticklabels([f"{s}\n(n={int((ok['style'] == s).sum())})" for s in order])
            ax.set_title(title)
        fig.suptitle("Prompt strategy variance (Manager step, frozen team report)", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_png, dpi=160)
        plt.close(fig)
        print(f"Chart written: {out_png}")
    except Exception as e:
        print(f"(chart skipped: {e})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--styles", default="react,cot,direct")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.analyze:
        analyze()
    else:
        run(a.repeats, [s.strip() for s in a.styles.split(",") if s.strip() in STYLES])
        analyze()
