# Supply Chain MAS — Comprehensive Audit & Clean Evaluation Report

**Generated**: 2026-11-24T19:17:04.832285
**Data Source**: `logs/runs.jsonl` (Audited without modification)

---

## 1. Dataset Integrity & Record Counts

| Metric | Count | Notes |
|---|---|---|
| Total Logged Records | **121** | Audited raw records in `runs.jsonl` |
| Unique Run IDs | **121** | Verified unique runs (Duplicates: 0) |
| Malformed JSON Lines | **0** | Complete JSON format integrity |
| Domain Experimental Runs | **121** | Production disruption scenarios |
| Test Harness Runs | **0** | Offline test suite invocations |
| **Strict Successes** | **41** | Strictly validated with Pydantic schema (`SupplyChainDecision`) + decision + no error |
| **Legacy Successes** | **0** | Nonempty decision + no error (prior to schema logging; unrecorded validation) |
| **Total Completed Decisions** | **41** | Total runs producing a validated decision without error |
| Total Failures / Errors | **74** | Classified in detail below |

> [!NOTE]
> **Schema Validation Audit**:
> - **Strict Successes (41)**: Runs verified against the Pydantic `SupplyChainDecision` output schema with `validation_outcome["passed"] == True`, non-empty final decision, and zero errors.
> - **Legacy Successes (0)**: Earlier runs producing valid decisions before schema outcome logging was enabled.
> - **Total Completed Decisions (41)**: Total runs generating verified supply chain disruption recovery recommendations.

---

## 2. Failure & Error Breakdown

| Error Category | Count | Primary Cause & Resolution |
|---|---|---|
| `rate_limit_quota` | **21** | Groq free-tier daily token limit (TPD / TPM cap reached during batch sweeps) |
| `agent_execution_error` | **0** | Tool timeouts or network aborts during multi-agent chaining |
| `guardrail_violation` | **0** | Adversarial input screened and blocked pre-execution |
| `missing_decision` | **0** | Incomplete runs where no final decision text was generated |

---

## 3. Architecture Pattern Comparison

> Empirical comparison across Sequential, Parallel, and Hierarchical architectures evaluated under identical disruption scenarios and persona constraints.

| Architecture | Attempts | Strict Success (Schema Validated) | Legacy Success (Unrecorded Schema) | Total Completed Decisions | Rate Limits | Agent Errors | Failure Rate | Mean Latency | Median Latency | Mean Judge |
|---|---|---|---|---|---|---|---|---|---|---|
| **Hierarchical** | 53 | 14 | 0 | 14 | 15 | 0 | 73.6% | 399.6s | 405.66s | 8.36/10 |
| **Parallel** | 34 | 14 | 0 | 14 | 3 | 0 | 58.8% | 284.97s | 278.36s | 8.93/10 |
| **Sequential** | 34 | 13 | 0 | 13 | 3 | 0 | 61.8% | 279.79s | 310.25s | 8.73/10 |

---

## 4. Latency & Performance by Persona (Successful Domain Runs)

| Target Persona | Sample Size (n) | Mean Latency | Std Dev | Median Latency | IQR (Q1–Q3) | Min Latency | Max Latency |
|---|---|---|---|---|---|---|---|
| **Aggressive Customer / Angry Executive** | 1 | 279.8s | 0.0s | 279.8s | [279.8s, 279.8s] | 279.8s | 279.8s |
| **CFO** | 15 | 281.41s | 70.01s | 276.75s | [267.56s, 317.37s] | 146.93s | 409.56s |
| **Operations Manager** | 14 | 341.8s | 90.86s | 317.73s | [286.61s, 355.75s] | 211.13s | 527.36s |
| **Supply Chain Planner** | 11 | 357.74s | 77.29s | 347.77s | [301.54s, 387.17s] | 261.41s | 509.75s |

---

## 5. Decision Quality (LLM-as-Judge) by Persona

> Evaluated using independent LLM-as-Judge rubric (scale 1–10). Evaluated only where judge scores were recorded (missing scores are not imputed as zero).

| Target Persona | Scored Runs (n) | Mean Judge Score | Median Score | Min Score | Max Score |
|---|---|---|---|---|---|
| **Aggressive Customer / Angry Executive** | 1 | 9.0 / 10 | 9.0 | 9.0 | 9.0 |
| **CFO** | 15 | 8.93 / 10 | 9.0 | 8.0 | 9.0 |
| **Operations Manager** | 14 | 8.5 / 10 | 9.0 | 6.5 | 9.0 |
| **Supply Chain Planner** | 11 | 8.5 / 10 | 9.0 | 5.5 | 9.0 |

---

## 6. Statistical Significance Analysis (Mann-Whitney U Rank-Sum Test)

> Non-parametric two-tailed test comparing latency distributions between target personas. Includes continuity and tie correction.

| Persona Comparison | n1 | n2 | U-Statistic | p-value | Significance (alpha = 0.05) | Finding |
|---|---|---|---|---|---|---|
| Aggressive Customer / Angry Executive vs CFO | 1 | 15 | 7.0 | 1.0 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |
| Aggressive Customer / Angry Executive vs Operations Manager | 1 | 14 | 3.0 | 0.4179 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |
| Aggressive Customer / Angry Executive vs Supply Chain Planner | 1 | 11 | 1.0 | 0.2466 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |
| CFO vs Operations Manager | 15 | 14 | 67.0 | 0.1017 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |
| CFO vs Supply Chain Planner | 15 | 11 | 38.0 | 0.0224 | **Significant (p < 0.05)** | Statistically distinguishable latency distribution. |
| Operations Manager vs Supply Chain Planner | 14 | 11 | 65.0 | 0.529 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |

---

## 7. Token Usage & Estimated Cost

> Token usage and API cost calculated from runs where Groq usage was returned. Missing entries are left as missing, never filled with zero.

- **Runs with verified token metrics**: 41 / 121 runs
- **Mean Total Tokens per Run**: 57,561 tokens
- **Mean Estimated Cost per Run**: $0.0131 USD (Groq commercial list price; free tier actual spend is $0.00)

### Token Usage and Cost Evidence by Architecture & Persona

| Architecture | Persona | Runs | Mean Total Tokens | Mean Est. Cost (USD) | Cost vs Baseline |
|---|---|---|---|---|---|
| **Hierarchical** | CFO | 5 | 38,185 | $0.0083 | -53.0% |
| **Hierarchical** | Operations Manager | 5 | 48,083 | $0.0105 | -40.6% |
| **Hierarchical** | Supply Chain Planner | 4 | 46,962 | $0.0102 | -41.8% |
| **Parallel** | CFO | 5 | 31,820 | $0.0072 | -59.3% |
| **Parallel** | Operations Manager | 5 | 30,500 | $0.0070 | -60.4% |
| **Parallel** | Supply Chain Planner | 4 | 32,320 | $0.0072 | -58.9% |
| **Sequential** | Aggressive Customer / Angry Executive | 1 | 31,075 | $0.0072 | -59.1% |
| **Sequential** | CFO | 5 | 209,012 | $0.0488 | +177.4% |
| **Sequential** | Operations Manager | 4 | 31,512 | $0.0070 | -60.1% |
| **Sequential** | Supply Chain Planner | 3 | 32,583 | $0.0074 | -58.2% |

---

## 8. Verified Output Artifacts

- Clean dataset: [`logs/dataset_clean_runs.csv`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/dataset_clean_runs.csv)
- Error report: [`logs/error_report.csv`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/error_report.csv)
- Per-agent latency table: [`logs/per_agent_latency.csv`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/per_agent_latency.csv)
- Heatmap: [`logs/agent_heatmap.png`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/agent_heatmap.png) (Missing values distinctly represented as gray `N/A`)
- Variance Boxplots: [`logs/boxplots.png`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/boxplots.png)
- Latency Distribution: [`logs/latency_boxplot.png`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/latency_boxplot.png)
- Bottleneck Flow: [`logs/pipeline_bottleneck.png`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/pipeline_bottleneck.png)
- Architecture Comparison: [`logs/architecture_comparison.png`](file:///C:/Users/drlsi/Downloads/supply-chain-mas-sequential/supply-chain-mas/logs/architecture_comparison.png)
