"""
Chart generation for the report's "Zooming Architectures" view
(MAS-as-a-whole view + per-agent bottleneck view).

Produces PNGs in logs/:
  1. pipeline_bottleneck.png  - the sequential pipeline drawn as a flow
  2. agent_latency_accuracy.png - bar chart of per-agent mean latency and judge score
  3. boxplots.png - latency and judge score distributions (median & IQR, non-outlier filtered)
  4. agent_heatmap.png - per-agent latency heat map with N/A support
  5. agent_tool_calls_heatmap.png - per-agent tool usage frequency heat map
  6. architecture_comparison.png - cross-pattern comparison (sequential vs parallel vs hierarchical)
"""

import matplotlib
matplotlib.use("Agg")  # headless-safe (no display needed)
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

PIPELINE_ORDER = [
    "Supplier Monitoring Agent",
    "Demand Forecasting Agent",
    "Inventory Agent",
    "Alternative Supplier Agent",
    "Cost Agent",
    "Manager Agent",
]


def _ordered(agent_summary):
    known = [a for a in PIPELINE_ORDER if a in agent_summary.index]
    extra = [a for a in agent_summary.index if a not in PIPELINE_ORDER]
    return agent_summary.loc[known + extra]


def pipeline_diagram(agent_summary, out_path="logs/pipeline_bottleneck.png"):
    df = _ordered(agent_summary)
    n = len(df)
    lat_col = "median_latency_sec" if "median_latency_sec" in df.columns else "mean_latency_sec"
    has_scores = "mean_judge_score" in df.columns and df["mean_judge_score"].notna().any()

    slowest = df[lat_col].idxmax()
    weakest = df["mean_judge_score"].idxmin() if has_scores else None

    fig, ax = plt.subplots(figsize=(3.0 * n, 3.6))
    ax.set_xlim(0, 3.0 * n)
    ax.set_ylim(0, 3.6)
    ax.axis("off")

    for i, (agent, row) in enumerate(df.iterrows()):
        x = 0.25 + i * 3.0
        edge, lw = "#334155", 1.5
        if agent == slowest:
            edge, lw = "#dc2626", 3.5
        elif agent == weakest:
            edge, lw = "#d97706", 3.5
        box = FancyBboxPatch((x, 0.8), 2.3, 1.9, boxstyle="round,pad=0.05",
                             fc="#f8fafc", ec=edge, lw=lw)
        ax.add_patch(box)
        ax.text(x + 1.15, 2.35, agent.replace(" Agent", ""), ha="center",
                va="center", fontsize=10, fontweight="bold")
        ax.text(x + 1.15, 1.75, f"{row[lat_col]:.1f}s", ha="center",
                va="center", fontsize=16, color="#0f172a")
        ax.text(x + 1.15, 1.35, "median latency", ha="center", va="center",
                fontsize=7.5, color="#64748b")
        if has_scores and row.get("mean_judge_score") == row.get("mean_judge_score"):
            ax.text(x + 1.15, 1.0, f"judge {row['mean_judge_score']:.1f}/10",
                    ha="center", va="center", fontsize=9, color="#0f172a")
        if i < n - 1:
            ax.annotate("", xy=(x + 3.2, 1.75), xytext=(x + 2.4, 1.75),
                        arrowprops=dict(arrowstyle="-|>", color="#334155",
                                        lw=1.8, mutation_scale=20))
            ax.text(x + 2.78, 1.95, "context", ha="center", fontsize=6.5,
                    color="#64748b")

    total = df[lat_col].sum()
    ax.text(0.25, 3.3, f"Sequential MAS - per-agent bottleneck view (sum of median latencies = {total:.0f}s)",
            fontsize=12, fontweight="bold")
    ax.text(0.25, 3.0, "Red outline = latency bottleneck"
            + ("   |   Amber outline = lowest judge score" if has_scores else ""),
            fontsize=8.5, color="#475569")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def agent_bars(agent_summary, by_persona, out_path="logs/agent_latency_accuracy.png"):
    df = _ordered(agent_summary)
    lat_col = "median_latency_sec" if "median_latency_sec" in df.columns else "mean_latency_sec"
    has_scores = "mean_judge_score" in df.columns and df["mean_judge_score"].notna().any()
    labels = [a.replace(" Agent", "") for a in df.index]

    ncols = 3 if has_scores else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.6))

    ax = axes[0]
    min_col = "min_latency_sec" if "min_latency_sec" in df.columns else lat_col
    max_col = "max_latency_sec" if "max_latency_sec" in df.columns else lat_col
    err_lo = (df[lat_col] - df[min_col]).clip(lower=0)
    err_hi = (df[max_col] - df[lat_col]).clip(lower=0)
    colors = ["#dc2626" if a == df[lat_col].idxmax() else "#3b82f6" for a in df.index]
    ax.bar(range(len(df)), df[lat_col], yerr=[err_lo, err_hi], capsize=3,
           color=colors, alpha=0.85, ecolor="#64748b")
    ax.set_xticks(range(len(df)))
    ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("seconds")
    ax.set_title("Per-agent latency (median + min/max range)")
    for i, v in enumerate(df[lat_col]):
        ax.text(i, v + (err_hi.iloc[i] if len(err_hi) else 0) + 0.3, f"{v:.1f}s",
                ha="center", fontsize=8.5, fontweight="bold")

    if has_scores:
        ax = axes[1]
        scores = df["mean_judge_score"].fillna(0)
        c_score = ["#d97706" if a == df["mean_judge_score"].idxmin() else "#10b981" for a in df.index]
        ax.bar(range(len(df)), scores, color=c_score, alpha=0.85)
        ax.set_xticks(range(len(df)))
        ax.set_xticklabels(labels, rotation=25, ha="right", fontsize=8.5)
        ax.set_ylabel("score /10")
        ax.set_ylim(0, 10.5)
        ax.set_title("Per-agent accuracy (LLM-as-Judge mean)")
        for i, v in enumerate(scores):
            ax.text(i, v + 0.2, f"{v:.1f}", ha="center", fontsize=8.5, fontweight="bold")

    ax = axes[-1]
    per_labels = [p.split(" - ")[0] for p in by_persona.index]
    ax.bar(range(len(by_persona)), by_persona["avg_latency_sec"], color="#6366f1", alpha=0.85)
    ax.set_xticks(range(len(by_persona)))
    ax.set_xticklabels(per_labels, rotation=15, ha="right", fontsize=8.5)
    ax.set_ylabel("seconds")
    ax.set_title("Total MAS latency by Persona")
    n_runs_list = by_persona["n_runs"] if "n_runs" in by_persona.columns else [None] * len(by_persona)
    for i, (v, n) in enumerate(zip(by_persona["avg_latency_sec"], n_runs_list)):
        label = f"{v:.1f}s" + (f"\n(n={int(n)})" if n is not None else "")
        ax.text(i, v + 0.5, label, ha="center", fontsize=8)

    fig.suptitle("Sequential MAS - latency and accuracy profile (outlier-filtered)", fontsize=11, y=0.99)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def _short_persona(p):
    return str(p).split(" - ")[0]


def plot_latency_boxplot(runs_df, out_path="logs/latency_boxplot.png"):
    """Plot latency boxplot across runs (convenience wrapper for clean_evaluation)."""
    import pandas as pd
    if isinstance(runs_df, list):
        runs_df = pd.DataFrame(runs_df)
    return latency_boxplots(runs_df, None, out_path=out_path)


def latency_boxplots(runs_df, agent_df, out_path="logs/boxplots.png"):
    """runs_df: non-outlier successful runs with persona, pipeline_latency, judge_score_val."""
    if runs_df is None or len(runs_df) == 0:
        return None
    personas = sorted(runs_df["persona"].unique())
    labels = [f"{_short_persona(p)}\n(n={int((runs_df['persona'] == p).sum())})" for p in personas]
    has_agents = agent_df is not None and len(agent_df) > 0
    ncols = 3 if has_agents else 2
    fig, axes = plt.subplots(1, ncols, figsize=(5.4 * ncols, 4.8))

    def _box(ax, data, tick_labels, color):
        bp = ax.boxplot(data, patch_artist=True, showmeans=True)
        ax.set_xticklabels(tick_labels)
        for patch in bp["boxes"]:
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        rng = np.random.default_rng(0)
        for i, d in enumerate(data, start=1):
            if len(d) > 0:
                ax.scatter(i + rng.uniform(-0.12, 0.12, len(d)), d, s=10, color="#1e293b", alpha=0.5, zorder=3)

    if "pipeline_latency" in runs_df.columns:
        lat_col = "pipeline_latency"
    elif "pipeline_latency_seconds" in runs_df.columns:
        lat_col = "pipeline_latency_seconds"
    elif "total_latency_seconds" in runs_df.columns:
        lat_col = "total_latency_seconds"
    else:
        lat_col = runs_df.columns[0]

    score_col = "judge_score_val" if "judge_score_val" in runs_df.columns else "judge_score"

    ax = axes[0]
    lat_data = [
        pd.to_numeric(runs_df.loc[runs_df["persona"] == p, lat_col], errors="coerce").dropna().values
        for p in personas
    ]
    _box(ax, lat_data, labels, "#3b82f6")
    ax.set_title("End-to-end latency per persona")
    ax.set_ylabel("seconds")

    ax = axes[1]
    def _extract_score(val):
        if isinstance(val, dict):
            return val.get("score")
        try:
            return float(val)
        except (ValueError, TypeError):
            return np.nan

    score_data = [
        pd.Series([_extract_score(v) for v in runs_df.loc[runs_df["persona"] == p, score_col]]).dropna().values
        for p in personas
    ]
    _box(ax, score_data, labels, "#10b981")
    ax.set_title("LLM-as-Judge score per persona")
    ax.set_ylabel("score /10")
    ax.set_ylim(0, 10.5)

    if has_agents:
        ax = axes[2]
        from_order = [a for a in PIPELINE_ORDER if a in set(agent_df["agent"])]
        data = [agent_df.loc[agent_df["agent"] == a, "latency_seconds"].dropna().values for a in from_order]
        _box(ax, data, [a.replace(" Agent", "").replace(" ", "\n") + f"\n(n={len(d)})"
                        for a, d in zip(from_order, data)], "#f59e0b")
        ax.set_title("Latency per agent")
        ax.set_ylabel("seconds")
        ax.tick_params(axis="x", labelsize=7)

    fig.suptitle("Variance across runs (box = IQR, line = median, triangle = mean)", fontsize=10, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def agent_heatmap(matrix, run_labels, out_path="logs/agent_heatmap.png"):
    """Heatmap 1: Per-agent latency (seconds).
    Missing values (np.nan) rendered distinctly in slate gray '#cbd5e1' with 'N/A'."""
    agents = [a for a in PIPELINE_ORDER][: matrix.shape[0]]
    n_runs = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(min(20.0, 2.5 + 0.32 * n_runs) if n_runs > 14
                                    else 1.0 + 0.95 * n_runs, 4.6))
    cmap = plt.cm.YlOrRd.copy()
    cmap.set_bad(color="#cbd5e1")

    masked_matrix = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked_matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(run_labels)))
    ax.set_xticklabels(run_labels, fontsize=8 if n_runs <= 25 else 6, rotation=0 if n_runs <= 25 else 90)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([a.replace(" Agent", "") for a in agents])

    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 10.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if n_runs <= 35:
                if np.isnan(v):
                    ax.text(j, i, "N/A", ha="center", va="center", fontsize=7, color="#64748b")
                else:
                    ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.6 * vmax else "#111827")

    ax.set_xlabel("run (oldest to newest)")
    ax.set_title("Per-agent latency heat map (seconds) - where execution time is spent\n"
                 "[Note: Gray cells (N/A) represent unrecorded / uninvoked steps, distinct from 0]",
                 fontsize=10, pad=10)
    cbar = fig.colorbar(im, ax=ax, label="seconds")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def agent_tool_calls_heatmap(matrix, run_labels, out_path="logs/agent_tool_calls_heatmap.png"):
    """Heatmap 2: Per-agent Tool Invocations Count across runs."""
    agents = [a for a in PIPELINE_ORDER][: matrix.shape[0]]
    n_runs = matrix.shape[1]
    fig, ax = plt.subplots(figsize=(min(20.0, 2.5 + 0.32 * n_runs) if n_runs > 14
                                    else 1.0 + 0.95 * n_runs, 4.6))
    cmap = plt.cm.Blues.copy()
    cmap.set_bad(color="#cbd5e1")

    masked_matrix = np.ma.masked_invalid(matrix)
    im = ax.imshow(masked_matrix, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(run_labels)))
    ax.set_xticklabels(run_labels, fontsize=8 if n_runs <= 25 else 6, rotation=0 if n_runs <= 25 else 90)
    ax.set_yticks(range(len(agents)))
    ax.set_yticklabels([a.replace(" Agent", "") for a in agents])

    vmax = np.nanmax(matrix) if not np.all(np.isnan(matrix)) else 5.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if n_runs <= 35:
                if np.isnan(v):
                    ax.text(j, i, "0", ha="center", va="center", fontsize=7, color="#94a3b8")
                else:
                    ax.text(j, i, f"{int(v)}", ha="center", va="center", fontsize=8,
                            color="white" if v > 0.6 * vmax else "#111827")

    ax.set_xlabel("run (oldest to newest)")
    ax.set_title("Per-agent Tool Invocations Count Heat Map across experiment runs",
                 fontsize=10, pad=10)
    cbar = fig.colorbar(im, ax=ax, label="tool calls")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def architecture_comparison(arch_df, out_path="logs/architecture_comparison.png"):
    names = list(arch_df.index)
    if not names:
        return None
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
    colors = ["#3b82f6", "#f59e0b", "#10b981", "#8b5cf6"][: len(names)]

    ax = axes[0]
    ax.bar(names, arch_df["mean_latency"], color=colors)
    ax.set_title("Mean latency (non-outlier runs)")
    ax.set_ylabel("seconds")
    for i, (lat, n) in enumerate(zip(arch_df["mean_latency"], arch_df["successes"])):
        ax.text(i, lat, f"{lat:.0f}s\n(n={int(n)})", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, arch_df["mean_latency"].max() * 1.25 if arch_df["mean_latency"].max() > 0 else 10)

    ax = axes[1]
    ax.bar(names, arch_df["mean_judge"], color=colors)
    ax.set_title("Mean judge score")
    ax.set_ylim(0, 10.8)
    for i, j in enumerate(arch_df["mean_judge"]):
        ax.text(i, j, f"{j:.2f}", ha="center", va="bottom", fontsize=9)

    ax = axes[2]
    ax.bar(names, arch_df["failure_rate_pct"], color=colors)
    ax.set_title("Failure rate (attempts that errored)")
    ax.set_ylabel("%")
    ax.set_ylim(0, 105)
    for i, (f, a) in enumerate(zip(arch_df["failure_rate_pct"], arch_df["attempts"])):
        ax.text(i, f, f"{f:.0f}%\n({int(a)} attempts)", ha="center", va="bottom", fontsize=9)

    fig.suptitle("Architecture pattern comparison (Sequential vs Parallel vs Hierarchical)", fontsize=9.5, y=0.995)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path
