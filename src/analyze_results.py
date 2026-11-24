"""
Summarizes logs/runs.jsonl to answer the core research questions:
1. Within the sequential MAS architecture, which agent is the bottleneck?
2. How consistent is decision quality and latency across personas?
3. How do Sequential, Parallel, and Hierarchical patterns compare on latency and reliability?

Handles outlier isolation (laptop sleeps / rate-limit hangs > 600s),
uses median and IQR for robust metrics, and exports clean datasets.

Usage:
    python -m src.analyze_results
"""

import os
import json
import numpy as np
import pandas as pd
from src.observability import load_all_runs
from src.charts import (pipeline_diagram, agent_bars, latency_boxplots,
                        agent_heatmap, agent_tool_calls_heatmap,
                        architecture_comparison, PIPELINE_ORDER)
from src.guardrails import guardrail_summary


OUTLIER_RUN_THRESHOLD = 600.0   # > 10 min is sleep/wait, not model inference
OUTLIER_STEP_THRESHOLD = 300.0  # > 5 min for a single agent step


def _ex(row, key):
    ex = row.get("extra")
    return ex.get(key) if isinstance(ex, dict) else None


def _real_steps(row):
    return [t for t in (row.get("agent_timings") or [])
            if not str(t.get("agent", "")).endswith("_full_run")]


def _calculate_pipeline_latency(row):
    # Crew wall-time if recorded, else sum of real agent steps, else total_latency_seconds
    crew_lat = _ex(row, "crew_latency_seconds")
    if crew_lat is not None and crew_lat == crew_lat and crew_lat > 0:
        return float(crew_lat)
    steps = _real_steps(row)
    if steps:
        return sum(float(t.get("latency_seconds", 0.0)) for t in steps)
    return float(row.get("total_latency_seconds", 0.0))


def main():
    runs = load_all_runs()
    if not runs:
        print("No runs logged yet in logs/runs.jsonl.")
        return

    df = pd.DataFrame(runs)
    def get_validation_status(r):
        structured_dec = _ex(r, "structured_decision")
        if isinstance(structured_dec, dict):
            try:
                from src.output_models import SupplyChainDecision
                s_obj = SupplyChainDecision.model_validate(structured_dec)
                return s_obj.validation_status
            except Exception:
                pass
        v = r.get("validation_outcome")
        if isinstance(v, dict):
            return "VALID" if v.get("passed") is True else "INVALID_LOGIC"
        return "unrecorded_legacy"

    df["validation_status"] = df.apply(get_validation_status, axis=1)
    df["success"] = df["error"].isna() & (df["validation_status"] == "VALID")
    df["judge_score_val"] = df.apply(
        lambda r: (r["judge_score"].get("score") if isinstance(r.get("judge_score"), dict) else (r["judge_score"] if isinstance(r.get("judge_score"), (int, float)) else None))
        if (r.get("final_decision") and str(r.get("final_decision")).strip() and get_validation_status(r) == "VALID") else None,
        axis=1
    )

    for _k in ("total_tokens", "estimated_cost_usd", "tools_via",
               "structuring_method", "crew_latency_seconds"):
        df[_k] = df.apply(lambda r, k=_k: _ex(r, k), axis=1)

    df["persona"] = df.apply(lambda r: r.get("persona") or _ex(r, "persona") or "Unknown", axis=1)
    df["scenario"] = df.apply(lambda r: r.get("scenario") or _ex(r, "scenario") or "unknown", axis=1)
    df["prompt_style"] = df.apply(lambda r: _ex(r, "prompt_style") or "react", axis=1)

    def get_error_cat(r):
        cat = _ex(r, "error_category")
        if cat and str(cat).lower() not in ("none", "", "nan"):
            return cat
        err = r.get("error")
        if not err or pd.isna(err):
            return None
        err_lower = str(err).lower()
        if any(k in err_lower for k in ["tpd", "tokens per day", "daily_limit", "quota", "exceeded your current quota"]):
            return "rate_limit_quota"
        elif any(k in err_lower for k in ["ratelimit", "rate_limit", "tpm", "tokens per minute", "too many requests"]):
            return "rate_limit_tpm"
        return "execution_error"

    df["error_category"] = df.apply(get_error_cat, axis=1)
    df["pipeline_latency"] = df.apply(_calculate_pipeline_latency, axis=1)

    # Outlier detection
    def is_outlier(row):
        if row["pipeline_latency"] > OUTLIER_RUN_THRESHOLD:
            return 1
        for s in _real_steps(row):
            if float(s.get("latency_seconds", 0.0)) > OUTLIER_STEP_THRESHOLD:
                return 1
        return 0

    df["outlier_flag"] = df.apply(is_outlier, axis=1)
    outlier_count = int(df["outlier_flag"].sum())

    print(f"Total runs logged: {len(df)}")
    print(f"Flagged latency outliers (sleep/hangs > {OUTLIER_RUN_THRESHOLD}s): {outlier_count}")
    print(f"Clean non-outlier runs: {len(df) - outlier_count}\n")

    # Architecture distribution & unequal sample warning
    arch_counts = df["architecture"].value_counts().to_dict()
    print("=== Architecture Sample Distribution ===")
    for a, cnt in arch_counts.items():
        clean_cnt = len(df[(df["architecture"] == a) & (df["outlier_flag"] == 0) & df["success"]])
        print(f"  - {a:15s}: {cnt:2d} total | {clean_cnt:2d} clean successful")

    if len(set(arch_counts.values())) > 1:
        print("\n[NOTE] Unequal sample sizes across architectures detected.")
        print("       (e.g. sequential vs parallel vs hierarchical).")
        print("       Report uses median/IQR to minimize skewness from sample size variations.")

    # Validation outcome analysis
    val_status = df["validation_outcome"].apply(
        lambda v: v.get("passed") if isinstance(v, dict) else None
    )
    passed_vals = val_status.sum() if val_status.notna().any() else 0
    print(f"\nStructured Output Validation: {passed_vals}/{len(df)} validated ({passed_vals/len(df)*100:.1f}%)")

    # Sequential MAS Analysis
    seq_all = df[df["architecture"] == "sequential"].copy()
    seq_clean = seq_all[(seq_all["outlier_flag"] == 0) & seq_all["success"]].copy()

    print(f"\n=== Sequential Runs (Clean Non-Outliers: {len(seq_clean)}) ===")
    by_persona = seq_clean.groupby("persona").agg(
        median_latency_sec=("pipeline_latency", "median"),
        avg_latency_sec=("pipeline_latency", "mean"),
        iqr_latency_sec=("pipeline_latency", lambda s: np.percentile(s, 75) - np.percentile(s, 25) if len(s) > 1 else 0),
        avg_judge_score=("judge_score_val", "mean"),
        n_runs=("run_id", "count"),
    ).round(2)
    print(by_persona)

    # Per-Agent Latency & Accuracy
    print("\n=== Sequential: Per-Agent Median Latency & Accuracy ===")
    agent_rows = []
    agent_tool_counts = []

    for _, row in seq_all.iterrows():
        tc_map = {}
        for tc in (row.get("tool_calls") or []):
            ag = tc.get("agent", "")
            tc_map[ag] = tc_map.get(ag, 0) + 1

        for t in _real_steps(row):
            ag_name = t["agent"]
            lat = float(t.get("latency_seconds", 0.0))
            is_step_outlier = 1 if lat > OUTLIER_STEP_THRESHOLD or row["outlier_flag"] == 1 else 0
            js = t.get("judge_score") or {}
            agent_rows.append({
                "run_id": row["run_id"],
                "timestamp": row["timestamp"],
                "persona": row["persona"],
                "scenario": row["scenario"],
                "agent": ag_name,
                "latency_seconds": lat,
                "judge_score": js.get("score") if isinstance(js, dict) else None,
                "outlier_flag": is_step_outlier,
                "tool_calls_count": tc_map.get(ag_name, 1),
            })

    agent_df = pd.DataFrame(agent_rows) if agent_rows else pd.DataFrame()

    if not agent_df.empty:
        # Filter clean rows for latency summary
        clean_agent_df = agent_df[agent_df["outlier_flag"] == 0]
        agg = {
            "median_latency_sec": ("latency_seconds", "median"),
            "mean_latency_sec": ("latency_seconds", "mean"),
            "iqr_latency_sec": ("latency_seconds", lambda s: np.percentile(s, 75) - np.percentile(s, 25) if len(s) > 1 else 0),
            "min_latency_sec": ("latency_seconds", "min"),
            "max_latency_sec": ("latency_seconds", "max"),
            "n_calls": ("latency_seconds", "count"),
        }
        if clean_agent_df["judge_score"].notna().any():
            agg["mean_judge_score"] = ("judge_score", "mean")

        agent_summary = clean_agent_df.groupby("agent").agg(**agg).round(2)
        agent_summary = agent_summary.sort_values("median_latency_sec", ascending=False)
        print(agent_summary)
        agent_summary.to_csv("logs/per_agent_latency.csv")
        print("\nClean per-agent metrics saved to logs/per_agent_latency.csv")

        # Generate diagrams
        p1 = pipeline_diagram(agent_summary)
        p2 = agent_bars(agent_summary, by_persona)
        p_box = latency_boxplots(seq_clean, clean_agent_df)

        # Generate dual heatmaps
        # Matrix 1: Latency
        clean_runs = seq_clean.sort_values("timestamp")
        clean_run_ids = list(clean_runs["run_id"])
        if clean_run_ids:
            lat_mat = np.full((len(PIPELINE_ORDER), len(clean_run_ids)), np.nan)
            tool_mat = np.full((len(PIPELINE_ORDER), len(clean_run_ids)), np.nan)

            for _, r in clean_agent_df.iterrows():
                rid = r["run_id"]
                ag = r["agent"]
                if rid in clean_run_ids and ag in PIPELINE_ORDER:
                    c_idx = clean_run_ids.index(rid)
                    r_idx = PIPELINE_ORDER.index(ag)
                    lat_mat[r_idx, c_idx] = r["latency_seconds"]
                    tool_mat[r_idx, c_idx] = r["tool_calls_count"]

            labels = [f"R{i+1}" for i in range(len(clean_run_ids))]
            p_heat1 = agent_heatmap(lat_mat, labels)
            p_heat2 = agent_tool_calls_heatmap(tool_mat, labels)
            print(f"Heat maps generated: {p_heat1}, {p_heat2}")

    # Cross-Architecture comparison (non-outliers)
    print("\n=== Architecture Pattern Comparison (Clean Non-Outliers) ===")
    clean_runs_all = df[df["outlier_flag"] == 0].copy()
    arch_agg = clean_runs_all.groupby("architecture").agg(
        attempts=("run_id", "count"),
        successes=("success", "sum"),
    )
    arch_agg["failure_rate_pct"] = (100 * (1 - arch_agg["successes"] / arch_agg["attempts"])).round(1)
    ok_runs = clean_runs_all[clean_runs_all["success"]]
    arch_agg["mean_latency"] = ok_runs.groupby("architecture")["pipeline_latency"].median().round(1)
    arch_agg["mean_judge"] = ok_runs.groupby("architecture")["judge_score_val"].mean().round(2)
    print(arch_agg)
    p_arch = architecture_comparison(arch_agg)
    if p_arch:
        print(f"Architecture comparison chart written: {p_arch}")

    # Export datasets
    cols = ["run_id", "timestamp", "architecture", "persona", "scenario", "success",
            "error_category", "outlier_flag", "pipeline_latency", "judge_score_val",
            "total_tokens", "estimated_cost_usd", "tools_via", "prompt_style"]
    avail_cols = [c for c in cols if c in df.columns]
    df[avail_cols].to_csv("logs/dataset_runs.csv", index=False)
    if not agent_df.empty:
        agent_df.to_csv("logs/dataset_agent_steps.csv", index=False)
    df.drop(columns=[c for c in ('extra',) if c in df.columns]).to_csv("logs/summary.csv", index=False)
    print("\nDatasets exported:")
    print("  - logs/dataset_runs.csv (one row per run, with outlier_flag)")
    print("  - logs/dataset_agent_steps.csv (one row per agent step, with outlier_flag)")
    print("  - logs/summary.csv (consolidated summary table)")

    print("\n=== Guardrail Audit Summary ===")
    print(guardrail_summary())


if __name__ == "__main__":
    main()
