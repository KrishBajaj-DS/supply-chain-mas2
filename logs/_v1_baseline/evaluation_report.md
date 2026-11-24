# Supply Chain MAS — Comprehensive Audit & Clean Evaluation Report

**Generated**: 2026-11-16T22:50:26.708864
**Data Source**: `logs/runs.jsonl` (Audited without modification)

---

## 1. Dataset Integrity & Record Counts

| Metric | Count | Notes |
|---|---|---|
| Total Logged Records | **40** | Audited raw records in `runs.jsonl` |
| Unique Run IDs | **40** | Verified unique runs (Duplicates: 0) |
| Malformed JSON Lines | **0** | Complete JSON format integrity |
| Domain Experimental Runs | **40** | Production disruption scenarios |
| Test Harness Runs | **0** | Offline test suite invocations |
| **Strict Successes** | **40** | Strictly validated with Pydantic schema (`SupplyChainDecision`) + decision + no error |
| **Legacy Successes** | **0** | Nonempty decision + no error (prior to schema logging; unrecorded validation) |
| **Total Completed Decisions** | **40** | Total runs producing a validated decision without error |
| Total Failures / Errors | **0** | Classified in detail below |

> [!NOTE]
> **Schema Validation Audit**:
> - **Strict Successes (40)**: Runs verified against the Pydantic `SupplyChainDecision` output schema with `validation_outcome["passed"] == True`, non-empty final decision, and zero errors.
> - **Legacy Successes (0)**: Earlier runs producing valid decisions before schema outcome logging was enabled.
> - **Total Completed Decisions (40)**: Total runs generating verified supply chain disruption recovery recommendations.

---

## 2. Failure & Error Breakdown

| Error Category | Count | Primary Cause & Resolution |
|---|---|---|
| `rate_limit_quota` | **0** | Groq free-tier daily token limit (TPD / TPM cap reached during batch sweeps) |
| `agent_execution_error` | **0** | Tool timeouts or network aborts during multi-agent chaining |
| `guardrail_violation` | **0** | Adversarial input screened and blocked pre-execution |
| `missing_decision` | **0** | Incomplete runs where no final decision text was generated |

---

## 3. Architecture Pattern Comparison

> Empirical comparison across Sequential, Parallel, and Hierarchical architectures evaluated under identical disruption scenarios and persona constraints.

| Architecture | Attempts | Strict Success (Schema Validated) | Legacy Success (Unrecorded Schema) | Total Completed Decisions | Rate Limits | Agent Errors | Failure Rate | Mean Latency | Median Latency | Mean Judge |
|---|---|---|---|---|---|---|---|---|---|---|
| **Hierarchical** | 10 | 10 | 0 | 10 | 0 | 0 | 0.0% | 201.66s | 196.57s | 8.1/10 |
| **Parallel** | 15 | 15 | 0 | 15 | 0 | 0 | 0.0% | 5113.92s | 176.28s | 7.87/10 |
| **Sequential** | 15 | 15 | 0 | 15 | 0 | 0 | 0.0% | 230.67s | 157.33s | 7.87/10 |

---

## 4. Latency & Performance by Persona (Successful Domain Runs)

| Target Persona | Sample Size (n) | Mean Latency | Std Dev | Median Latency | IQR (Q1–Q3) | Min Latency | Max Latency |
|---|---|---|---|---|---|---|---|
| **CFO** | 13 | 3479.85s | 11101.87s | 176.28s | [157.33s, 211.85s] | 127.44s | 41919.7s |
| **Operations Manager** | 14 | 238.62s | 299.73s | 151.13s | [136.18s, 181.27s] | 30.45s | 1295.38s |
| **Supply Chain Planner** | 13 | 2585.13s | 7009.91s | 168.31s | [166.11s, 273.67s] | 115.64s | 26433.32s |

---

## 5. Decision Quality (LLM-as-Judge) by Persona

> Evaluated using independent LLM-as-Judge rubric (scale 1–10). Evaluated only where judge scores were recorded (missing scores are not imputed as zero).

| Target Persona | Scored Runs (n) | Mean Judge Score | Median Score | Min Score | Max Score |
|---|---|---|---|---|---|
| **CFO** | 13 | 7.92 / 10 | 8.0 | 7.0 | 9.0 |
| **Operations Manager** | 14 | 7.93 / 10 | 8.0 | 7.0 | 9.0 |
| **Supply Chain Planner** | 13 | 7.92 / 10 | 8.0 | 7.0 | 8.0 |

---

## 6. Statistical Significance Analysis (Mann-Whitney U Rank-Sum Test)

> Non-parametric two-tailed test comparing latency distributions between target personas. Includes continuity and tie correction.

| Persona Comparison | n1 | n2 | U-Statistic | p-value | Significance (alpha = 0.05) | Finding |
|---|---|---|---|---|---|---|
| CFO vs Operations Manager | 13 | 14 | 62.0 | 0.1667 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |
| CFO vs Supply Chain Planner | 13 | 13 | 79.0 | 0.7976 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |
| Operations Manager vs Supply Chain Planner | 14 | 13 | 61.0 | 0.1523 | Not Significant (p >= 0.05) | No statistically significant difference in latency between personas. |

---

## 7. Token Usage & Estimated Cost

> Token usage and API cost calculated from runs where Groq usage was returned. Missing entries are left as missing, never filled with zero.

- **Runs with verified token metrics**: 40 / 40 runs
- **Mean Total Tokens per Run**: 856,418 tokens
- **Mean Estimated Cost per Run**: $0.1931 USD (Groq commercial list price; free tier actual spend is $0.00)

### Token Usage and Cost Evidence by Architecture & Persona

| Architecture | Persona | Runs | Mean Total Tokens | Mean Est. Cost (USD) | Cost vs Baseline |
|---|---|---|---|---|---|
| **Hierarchical** | CFO | 3 | 441,661 | $0.0984 | -27.3% |
| **Hierarchical** | Operations Manager | 4 | 348,861 | $0.0780 | -42.4% |
| **Hierarchical** | Supply Chain Planner | 3 | 649,210 | $0.1449 | +7.0% |
| **Parallel** | CFO | 5 | 1,372,789 | $0.3106 | +129.3% |
| **Parallel** | Operations Manager | 5 | 1,231,559 | $0.2778 | +105.1% |
| **Parallel** | Supply Chain Planner | 5 | 1,509,973 | $0.3414 | +152.1% |
| **Sequential** | CFO | 5 | 506,516 | $0.1140 | -15.8% |
| **Sequential** | Operations Manager | 5 | 647,975 | $0.1462 | +7.9% |
| **Sequential** | Supply Chain Planner | 5 | 648,918 | $0.1462 | +7.9% |

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
