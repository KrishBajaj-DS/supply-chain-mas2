"""
Regenerates all evaluation reports, summary metrics, and visualizations
from the recorded CSV dataset.
Usage:
    python regenerate_reports.py
"""
import sys
import subprocess
from pathlib import Path

def main():
    root = Path(__file__).resolve().parent
    print(f"Running report and chart regeneration in {root}...")
    cmd = [sys.executable, "-m", "src.clean_evaluation"]
    res = subprocess.run(cmd, cwd=root)
    if res.returncode == 0:
        print("\nAll charts, summary tables, and datasets regenerated successfully.")
        print("Generated files:")
        print("  - logs/evaluation_report.md (Statistical significance, error breakdown, latency distributions)")
        print("  - logs/dataset_clean_runs.csv (Clean validated dataset)")
        print("  - logs/error_report.csv (Detailed failure & quota error audit)")
        print("  - logs/summary.csv")
        print("  - logs/dataset_runs.csv")
        print("  - logs/dataset_agent_steps.csv")
        print("  - logs/per_agent_latency.csv")
        print("  - logs/agent_heatmap.png (with missing values distinctly rendered)")
        print("  - logs/boxplots.png")
        print("  - logs/latency_boxplot.png")
        print("  - logs/pipeline_bottleneck.png")
        print("  - logs/architecture_comparison.png")
        print("  - logs/agent_latency_accuracy.png")
    else:
        print(f"Failed to regenerate reports. Exit code: {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
