# Design Notes — Supply Chain Disruption Management MAS

This doc maps the project directly onto the course rubric (`Project_Guidelines.pdf`)
providing rigorous, citable evidence for each requirement and explaining all architectural choices.

---

## 1. Goal, Environment, KPI, Personas (Ch.1)

- **Goal**: Given a supply-chain disruption (e.g. port congestion, factory fire, customs hold), autonomously orchestrate specialist agents to evaluate inventory, forecast demand, screen alternative vendors, audit vendor compliance, and produce ONE logically consistent, cost-optimal resolution (`wait` / `switch_supplier` / `expedite_shipment`).
- **Environment**: Partially observable (agents observe through tool outputs and shared blackboard state), episodic per run, static within an execution cycle, discrete action space, multi-agent cooperative architecture.
- **Percepts & Domain Space**: Shipment status, linear trend demand forecasting, warehouse inventory balances, supplier knowledge base (RAG), cost impact calculators, and vendor regulatory compliance audits.
- **Key Performance Indicators (KPIs)**:
  1. *Decision Quality*: 10-point rubric score (Shortfall arithmetic, action consistency, cost-for-quantity, persona alignment, compliance audit integration).
  2. *Operational Efficiency*: Wall-clock pipeline latency and per-agent latency distribution (measured via median and IQR).
  3. *Cost & Quota Safety*: Token accounting and estimated cost in USD ($0 on Groq free tier).
- **Target Personas (`src/scenarios.py`)**:
  - *Operations Manager*: Prioritizes factory continuity and low stockout risk; accepts premium options only when shortfall > 0.
  - *CFO*: Strictly minimizes financial outlay; prefers `wait` on zero shortfall, selects lowest-cost bridge option when shortfall > 0.
  - *Supply Chain Planner*: Prioritizes dual-sourcing reliability and vendor compliance; defaults to `wait` with monitoring checkpoints when buffer surplus exists.
  - *Angry Executive Personas (`angry_exec_*`)*: Characters exhibiting operational frustration/abusive language; screened by guardrails which sanitize profane terms with `[REDACTED]` while executing legitimate requests.

---

## 2. Decision Logic & Pydantic Schema Enforcement (Phases 1 & 3)

### Decision Logic Rules (`src/tasks.py`, `src/agents.py`):
1. **Shortfall Arithmetic**:
   $$\text{shortfall} = \max(0, \text{forecasted\_demand\_over\_delay} - \text{current\_stock})$$
   Agents are strictly forbidden from describing forecast demand as a "shortfall" when stock exceeds demand.
2. **Action Consistency**:
   - If $\text{shortfall} = 0$: Default action MUST be `"wait"` (or monitor). A switch/expedite is rejected unless the justification quantifies an explicit risk (e.g. buffer runway < 2 days).
   - If $\text{shortfall} > 0$: Action MUST be `"expedite_shipment"`, `"switch_supplier"`, or `"dual_source"` sized to close the gap.
3. **Cost-for-Quantity**:
   Options evaluated calculate total cost and cost delta for the quantity needed (e.g. 800 units expedited), NOT the entire bulk shipment (e.g. 3,000 units).
4. **Shipment Fate & Overstock**:
   The decision report explicitly accounts for the in-transit shipment (whether kept, cancelled, or delayed) and quantifies post-arrival overstock risk.

### Post-Generation Logic Validator (`src/output_models.py`):
Pydantic `@model_validator` executes automatic consistency audits on generated decisions:
- Flags `validation_status = "INVALID_LOGIC"` if zero shortfall claims a shortfall exists or cites "eliminating shortfall".
- Flags invalid logic if switch/expedite is chosen on zero shortfall without a quantified risk explanation.

---

## 3. Tooling & Model Context Protocol (MCP) (Ch.2/3)

### Specialist Tools (`src/tools.py`):
1. `check_shipment_status`: Retrieves tracking, carrier, and disruption reason.
2. `forecast_demand`: Computes projected consumption over the disruption window.
3. `check_inventory`: Evaluates stock balance, calculates buffer runway, and identifies shortfall.
4. `calculate_cost_impact`: Calculates unit-level and order-level cost deltas including expedited air freight premiums.
5. `search_alternative_suppliers`: Semantic RAG retriever querying a 5-supplier knowledge base (`src/supplier_kb.py`) with cosine similarity.

### Model Context Protocol (MCP) (`src/mcp_server.py`):
Exposes the 4 core supply-chain tools over the official **Model Context Protocol (MCP)** using stdio JSON-RPC transport. CrewAI connects via `MCPServerAdapter` (`src/run_experiments.py`), ensuring tools are invoked over standard external protocol interfaces with automatic subprocess lifecycle management via Python `ExitStack`.

---

## 4. Messaging Protocol: Blackboard Architecture & P2P Comparison (Ch.3/6)

- **Blackboard Shared State (`src/blackboard.py`)**:
  - Thread-safe central blackboard partitioned into validated sections: `supplier_status`, `demand_forecast`, `inventory_projection`, `alternative_suppliers`, `cost_analysis`, and `compliance_audit`.
  - Specialists publish findings dynamically via `post_to_blackboard`; downstream agents read state via `read_blackboard`.
  - Run-scoped isolation via `reset_blackboard()` prevents cross-run contamination.
- **Empirical Peer-to-Peer Comparison (`src/compare_architectures.py`)**:
  - Compares Blackboard against point-to-point message passing (`src/peer_to_peer.py`).
  - **Blackboard**: Retains **100.0%** upstream domain context at the Manager decision point.
  - **Peer-to-Peer**: Drops to **20.0%** context retention as intermediate shipment and inventory details decay across sequential handoffs.

---

## 5. Dual Framework & A2A Protocol Interoperability (Criteria 6 & 7)

### Real Google ADK Implementation (`src/adk_agent.py`):
- Uses the official `google-adk>=2.14.0` package.
- Autonomous `GoogleADKComplianceAgent` constructed with `google.adk.Agent`, `google.adk.Runner`, `InMemorySessionService`, and function tools (`audit_supplier_vulnerability`, `evaluate_vendor_network`).
- Strictly zero foreign agentic frameworks (no LangChain, AutoGen, or LangGraph).

### Agent-to-Agent (A2A) Protocol (`src/a2a_protocol.py`, `src/a2a_crew_bridge.py`):
- FIPA-compliant message envelopes (`A2AMessage`) with unique conversation IDs, timestamps, sender/recipient framework tags, and performatives (`REQUEST`, `INFORM`, `FAILURE`).
- Router (`A2ABus`) dispatches cross-framework queries between CrewAI and Google ADK.
- CrewAI agents invoke the ADK agent to audit supplier risk (vulnerability index, ISO 9001 compliance, single-source risk), automatically posting findings to `blackboard['compliance_audit']`.

---

## 6. Three Architecture Patterns Evaluated (Criterion 4)

1. **Sequential Architecture (`src/crew_sequential.py`)**:
   Deterministic pipeline: Supplier Monitoring $\to$ Demand Forecasting $\to$ Inventory $\to$ Alternative Supplier $\to$ Cost $\to$ Manager. Highest reliability and lowest token overhead.
2. **Parallel Fan-out / Fan-in Architecture (`src/crew_parallel.py`)**:
   Demand Forecasting and Alternative Supplier sourcing execute concurrently (`async_execution=True`) in a fan-out pattern, then fan-in to Inventory and Cost agents.
3. **Hierarchical Architecture (`src/crew_hierarchical.py`)**:
   CrewAI `Process.hierarchical` with dedicated `manager_llm`. Manager autonomously delegates sub-tasks to specialist coworkers and coordinates the synthesis.

---

## 7. Security Guardrails & Input Sanitisation (Criterion 8)

Multi-tier guardrail system implemented in pure Python (`src/guardrails.py`):
1. **Pre-Execution Input Screening**:
   - *Prompt Injections*: Regex screening intercepts instructions like `"ignore previous instructions"`, `"act as jailbroken"`, or `"reveal system prompt"`, raising `GuardrailViolation` before any agent or LLM executes.
   - *Input Hygiene*: Type checks, empty inputs, minimum length (<10 chars), and maximum length (>2000 chars).
   - *Abusive Language & Persona Sanitisation*: Purely abusive input is blocked. Legitimate domain requests with aggressive persona language (e.g. angry executive character) are sanitized using `[REDACTED]` replacement, enabling safe processing without crash.
2. **Tool Output Validation (`@guarded`)**:
   Validates tool outputs against domain rules (e.g. negative ordered units, implausible lead times). Replaces invalid data with a safe warning to prevent poisoned downstream reasoning.
3. **Audit Trail**:
   All security checks recorded to `logs/guardrail_events.jsonl` with timestamps and violation categories.

---

## 8. AI Observability, Evaluation & Outlier Isolation (Criteria 5, 9, 10)

- **RunTracer (`src/observability.py`)**:
  Appends execution records to `logs/runs.jsonl`: `run_id`, timestamp, architecture, persona, scenario, per-agent latencies, tool call metadata, provenance trace, validation status, and judge scores.
- **Per-Run Token Isolation (`src/cost.py`)**:
  Delta-based accounting computes run-scoped token usage and resets baselines, preventing cumulative token inflation across runs.
- **Latency Outlier Isolation (`src/analyze_results.py`, `scripts/repair_logs.py`)**:
  Identifies runs where laptop sleep or rate-limit waits distorted latency (`pipeline_latency > 600s` or any step `> 300s`). Flags outliers via `outlier_flag` and computes summaries (`per_agent_latency.csv`, `boxplots.png`) using median and interquartile range (IQR).
- **Dual Heatmaps (`src/charts.py`)**:
  1. *Per-agent Latency Heatmap*: Color-coded wall-clock time per agent across runs, with missing values (e.g. hierarchical uninvoked agents) rendered distinctly in slate gray `N/A`.
  2. *Tool Call Count Heatmap*: Tracks tool invocation frequencies per agent.
