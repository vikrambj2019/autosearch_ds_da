# Multi-Agent Analytics System

Natural language analytics platform built on the Claude Agent SDK. Uses a **2-pass deterministic architecture** — Python computes all numbers, the LLM only writes narrative.

## Quick Start

```bash
# --data is required — pass the path to your CSV file

# Descriptive question (panel data)
python3 orchestrator.py --data Data/pannel_data.csv "What is the distribution of MONTHLY_TOTAL_COST by CURRENTLY_ACTIVE?"

# Diagnostic question
python3 orchestrator.py --data Data/pannel_data.csv "Is there a significant difference in cost between male and female?"

# ML model build (cross-sectional data)
python3 orchestrator.py --data Data/raw_data.csv "Build me a classification model predicting TARGET_HIGH_COST_FLAG"

# Period comparison (panel data)
python3 orchestrator.py --data Data/pannel_data.csv "Compare July vs September costs"

# Quick mode (Pass 1 only, skip depth analyses)
python3 orchestrator.py --data Data/pannel_data.csv --quick "How many members are there?"
```

## Architecture

### 2-Pass Deterministic Engine (analytics questions)

```
Question → Route → Profile → Extract Params
  → Pass 1: Deterministic primary analysis (Python, no LLM)
  → Pass 2: 2 deterministic depth analyses (Python, no LLM)
  → Narrate: Single LLM call turns all JSON results into a report
```

The LLM NEVER computes numbers. Every number comes from `tools/analytics.py`. The LLM only writes English narrative from pre-computed JSON, including a pre-computed `comparison.direction` field so it cannot get higher/lower wrong.

### ML Workflow — Autoresearch Pattern (classification questions)

```
Question → Route to ML
  → Round 1: Agent profiles data + generates baseline pipeline code
  → Orchestrator runs pipeline, measures composite_score
  → Round 2+: Agent gets best code + experiment log, makes ONE change
  → Orchestrator runs, compares: KEEP (if better) or DISCARD (if worse)
  → Repeat until: score >= 0.65, plateau, or max 6 rounds
  → Narrate: Single LLM call writes final report from experiment log
```

Inspired by [karpathy/autoresearch](https://github.com/karpathy/autoresearch): the orchestrator controls the experiment loop with keep/discard logic. The agent generates code but does NOT execute it — the orchestrator runs the pipeline and decides whether to keep or revert.

## Codebase Structure

```
orchestrator.py          — Entry point: 2-pass engine + ML workflow
agents/
  definitions.py         — Skill discovery, routing, AgentDefinition builder
tools/
  analytics.py           — Deterministic computation layer (7 functions)
  data_tools.py          — MCP tool handlers: profile, run_code, validate_cols, analytics
  ml_tools.py            — MCP tool handlers: run_pipeline, score_metrics
  servers.py             — MCP server wiring (data server + ml server)
hooks/
  safety.py              — PreToolUse Bash safety guardrails
registry/
  descriptive/           — "What happened?" skill (prompt + manifest)
  diagnostic/            — "Why did it happen?" skill (prompt + manifest)
  classification/        — ML model building skill (+ template.py)
  critic/                — Sanity check + follow-up generation (used by ML workflow)
  reporter/              — Insight report compilation (used by ML workflow)
  evaluator/             — LLM-as-judge quality scoring (used by ML workflow)
company_context/
  domain_knowledge.md    — Healthcare cost analytics context
  metrics_definitions.md — Column/metric definitions
Data/                    — Local test data (panel + cross-sectional)
output/                  — Generated reports (timestamped markdown)
SPEC_2026_03_12.md       — Full system specification
```

## Key Concepts

- **Grain awareness**: Auto-detects panel (entity x time) vs cross-sectional data; all tools aggregate correctly
- **Deterministic analytics**: `tools/analytics.py` has 7 functions — `distribution_by_category`, `trend_over_time`, `group_comparison`, `correlation_analysis`, `summary_stats`, `entity_counts`, `period_comparison` — each grain-aware with automatic panel collapse
- **Period comparison**: Entity-level diff between two time periods with auto driver analysis across all categorical columns, cohort matching, transition matrix, and top movers
- **High-cardinality handling**: Pareto-aware bucketing — keeps categories covering 95% of metric value OR top 10 by count (whichever keeps more), so high-value segments aren't buried in "Other"
- **Pre-computed direction**: Every comparison includes a `comparison.direction` field the LLM must use verbatim
- **Skill registry**: Add new skills by creating a directory under `registry/` with manifest.json + prompt.md
- **Company context**: Domain knowledge injected into all agent prompts from `company_context/`
- **Parameter extraction**: LLM-based extraction resolves natural language to exact column names and time periods; keyword fallback if LLM fails
- **Depth analysis rules**: Rule-based selection of 2 follow-up analyses based on primary analysis type

## Analytics Tools (tools/analytics.py)

| Function | Use case |
|---|---|
| `distribution_by_category(df, metric, category)` | Distribution/breakdown of metric by category |
| `trend_over_time(df, metric, stratify_by=)` | Time series trend with regression stats |
| `group_comparison(df, metric, group_col)` | Statistical test between groups (t-test, Mann-Whitney, Cohen's d) |
| `correlation_analysis(df, metric_a, metric_b)` | Pearson/Spearman correlation |
| `summary_stats(df, metric)` | Percentiles, skewness, zero %, total |
| `entity_counts(df, group_col=)` | Unique entity counts, optionally grouped |
| `period_comparison(df, metric, period_a=, period_b=)` | Entity-level diff between periods + auto drivers |

## Dependencies

- `claude-agent-sdk` (pip installed, not from local src/)
- `pandas`, `numpy`, `scipy`
- `lightgbm`, `xgboost`, `shap`, `scikit-learn` (for ML skill)
- `tabulate` (for markdown table output)
