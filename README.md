# Supply Chain Disruption Management Multi-Agent System (MAS)

Agentic AI course project. An enterprise-grade multi-agent system that autonomously responds to supply-chain shipment disruptions. Built using **CrewAI** and **Google ADK** (`google-adk`), communicating via the standardized **A2A Protocol**, instrumented with real per-agent observability, security guardrails, and deterministic evaluation.

See `docs/DESIGN_NOTES.md` for the full rubric-by-rubric write-up.

---

## Course Criteria Compliance Matrix (1–10)

| # | Course Criterion | Implementation & Evidence Files | Verification Command | Status |
|---|---|---|---|---|
| **1** | **Blackboard Shared State & Architecture Comparison** | Central shared state in `src/blackboard.py` (typed sections: `supplier_status`, `demand_forecast`, `inventory_projection`, `alternative_suppliers`, `cost_analysis`, `compliance_audit`). Point-to-point comparison in `src/peer_to_peer.py` & `src/compare_architectures.py`. | `python -m src.compare_architectures` | **PASS** |
| **2** | **External Tools via Model Context Protocol (MCP)** | Stdio RPC server in `src/mcp_server.py` exposing 4 tools (`check_shipment_status`, `forecast_demand`, `check_inventory`, `calculate_cost_impact`). Integrated into CrewAI via `MCPServerAdapter` in `src/run_experiments.py`. | `python -m src.mcp_demo`<br>`python -m src.mcp_agent_demo --dry` | **PASS** |
| **3** | **Pydantic Structured Output & Logic Validator** | `SupplyChainDecision` in `src/output_models.py` with `@model_validator` enforcing zero-shortfall wait logic, shortfall arithmetic, and quantity-specific cost accounting. | `python -m unittest tests/test_all_requirements.py` | **PASS** |
| **4** | **Three Architecture Patterns Evaluated** | Sequential (`src/crew_sequential.py`), Parallel fan-out/fan-in (`src/crew_parallel.py`), and Hierarchical delegation (`src/crew_hierarchical.py`) with common CLI in `src/run_architecture.py`. | `python -m src.run_architecture --architecture sequential --dry-run` | **PASS** |
| **5** | **Task Monitoring, Callbacks & Dual Heatmaps** | Step callbacks in `src/observability.py` (`make_step_callback`), logging intermediate steps. Dual heatmaps in `src/charts.py`: (1) Latency per agent with gray N/A, (2) Tool invocations count per agent. | `python -m src.analyze_results`<br>`logs/agent_heatmap.png` | **PASS** |
| **6** | **Agent-to-Agent (A2A) Protocol Communication** | Standardized FIPA-style envelope and router in `src/a2a_protocol.py`. Cross-framework bridge in `src/a2a_crew_bridge.py` linking CrewAI and Google ADK. | `python -m unittest tests.test_all_requirements.TestRequirement6A2AProtocol` | **PASS** |
| **7** | **Dual Framework: CrewAI + Real Google ADK** | Official `google-adk>=2.14.0` package. Autonomous `GoogleADKComplianceAgent` in `src/adk_agent.py` using `adk.Agent`, `adk.Runner`, `InMemorySessionService`, and function tools. Strictly zero unapproved frameworks. | `python -m unittest tests.test_all_requirements.TestRequirement7GoogleADKDualFramework` | **PASS** |
| **8** | **Security Guardrails, Personas & Input Sanitisation** | Pre-execution screening in `src/guardrails.py` for prompt injections, type errors, and tool corruption. Handles aggressive/hostile executive personas (`angry_exec_*`) via `[REDACTED]` sanitisation. | `python -m src.guardrail_demo` | **PASS** |
| **9** | **AI Observability, Auditability & Lineage** | `RunTracer` in `src/observability.py` logging per-run unique IDs, wall-clock latencies, tool metadata, validation status, and blackboard provenance in `logs/runs.jsonl`. | `logs/runs.jsonl`<br>`logs/guardrail_events.jsonl` | **PASS** |
| **10** | **Evaluation Dataset, Prompt Variance & Boxplots** | 40+ runs evaluated across ReAct, CoT, and Direct styles (`src/prompt_ablation.py`). Token resets in `src/cost.py`. Outlier detection (>600s) and boxplots with median/IQR in `src/charts.py`. | `python regenerate_reports.py`<br>`logs/boxplots.png` | **PASS** |

---

## Setup

```bash
# 1. Activate project virtual environment
source venv/bin/activate       # On Linux/macOS
# or on Windows PowerShell:
.\venv\Scripts\Activate.ps1

# 2. Install dependencies (includes crewai and google-adk)
pip install -r requirements.txt

# 3. Environment configuration (placeholders provided; offline modes consume 0 quota)
cp .env.example .env
```

---

## Verification & Execution Commands (Sequential Guide)

### 1. Instant Offline Demonstration (0 Quota, < 1s)
Replays full MAS execution: guardrail check, blackboard progression, A2A communication with Google ADK, and final Pydantic decision:
```bash
python -m src.demo_replay --scenario widget_a
python -m src.demo_replay --scenario sensor_b
python -m src.demo_replay --scenario cable_harness
# Or replay all three:
python -m src.demo_replay --all
```

### 2. Comprehensive Offline Unit Test Suite (45 Tests, 100% Pass)
Runs the complete test suite verifying all 10 course requirements offline:
```bash
python -m unittest tests/test_all_requirements.py -v
```

### 3. Log Repair & Dataset Synchronization
Deterministic offline tool to audit, clean, and synchronize run logs and Pydantic decisions:
```bash
python -m scripts.repair_logs
```

### 4. LLM-as-Judge 5-Criteria Rubric Backfill
Scores all runs on shortfall arithmetic, action consistency, cost-for-quantity, persona alignment, and ADK compliance:
```bash
python -m src.judge --backfill
```

### 5. Prompt Ablation Evaluation (ReAct vs. CoT vs. Direct)
Analyzes Manager reasoning variants and outputs prompt variance metrics:
```bash
python -m src.prompt_ablation --analyze
```

### 6. Full Report & Visualization Regeneration
Regenerates all evaluation figures, summary metrics, and LaTeX-ready markdown reports:
```bash
python regenerate_reports.py
python -m src.analyze_results
```

Generated Output Files in `logs/`:
- `logs/dataset_runs.csv`: 40 runs with linked `run_id`, `scenario`, `prompt_style`, `outlier_flag`, `pipeline_latency`.
- `logs/dataset_agent_steps.csv`: 195 agent step latency records.
- `logs/per_agent_latency.csv`: Clean median and IQR latency per agent.
- `logs/decisions_structured.jsonl`: 40 schema-validated structured decisions.
- `logs/agent_heatmap.png`: Per-agent latency heatmap with gray N/A representation.
- `logs/agent_tool_calls_heatmap.png`: Per-agent tool invocations count heatmap.
- `logs/boxplots.png`: End-to-end latency and judge score boxplots (median and IQR).
- `logs/architecture_comparison.png`: Cross-pattern comparison across Sequential, Parallel, and Hierarchical.
- `logs/pipeline_bottleneck.png`: Bottleneck flow diagram with latency and accuracy indicators.

### 7. Interactive Streamlit Dashboard
```bash
streamlit run app.py
```
Explore the interactive architecture playground, A2A protocol dialogue inspect tool, and compliance matrix.
