# Multi-Agent Analytics System — Architecture Guide

This document explains how the system works end-to-end in plain language. No diagram required.

---

## Table of Contents

1. [What the system does](#1-what-the-system-does)
2. [The core design principle](#2-the-core-design-principle)
3. [How a question flows through the system](#3-how-a-question-flows-through-the-system)
4. [The two workflows](#4-the-two-workflows)
   - [Analytics workflow (2-pass engine)](#41-analytics-workflow-2-pass-engine)
   - [ML workflow (autoresearch loop)](#42-ml-workflow-autoresearch-loop)
5. [File-by-file guide](#5-file-by-file-guide)
6. [The analytics tools in detail](#6-the-analytics-tools-in-detail)
7. [How routing works](#7-how-routing-works)
8. [Safety guardrails](#8-safety-guardrails)
9. [Data layer](#9-data-layer)
10. [Cost and performance](#10-cost-and-performance)
11. [Key design decisions and why](#11-key-design-decisions-and-why)
12. [Common usage examples](#12-common-usage-examples)

---

## 1. What the system does

You give it a plain-English question and a CSV file. It answers the question with a grounded, written report.

```
python3 orchestrator.py --data Data/pannel_data.csv \
  "What is the distribution of monthly cost by active status?"
```

Internally, the system:
- Figures out what kind of analysis your question is asking for
- Runs deterministic Python code to compute every number
- Feeds those pre-computed numbers to an LLM (Claude), which writes the report in plain English
- Saves the report to `output/`

The LLM never touches arithmetic. It only writes English sentences from numbers that Python already computed.

---

## 2. The core design principle

**The LLM narrates; Python computes.**

Every number in the final report comes from a deterministic Python function. The LLM receives a JSON blob of pre-computed results and translates it into readable prose. This eliminates a whole class of hallucination: wrong counts, wrong percentages, wrong directions ("Group A is higher" when actually Group B is higher).

Two concrete mechanisms enforce this:

- **Pre-computed direction field.** Every comparison result includes a `comparison.direction` string like `"INACTIVE members have 2.3x higher cost than ACTIVE members"`. The LLM is instructed to copy this verbatim rather than re-derive it.
- **Upfront grain context.** Before the LLM runs, the system profiles the dataset and injects facts (row count, entity count, time periods, column names) directly into the prompt. The LLM cannot hallucinate "12 months of data" when the prompt already states "3 months: Jul, Aug, Sep 2025."

---

## 3. How a question flows through the system

At the highest level, every question goes through four steps:

**Step 1 — Route.** The question text is scored against a library of trigger phrases in the skill manifests. The highest-scoring skill is selected (e.g., `descriptive`, `diagnostic`, or `classification`).

**Step 2 — Extract parameters.** A single LLM call (Sonnet, `max_turns=1`) receives the question, column names, column types, categorical values, and grain metadata. It returns a structured JSON with: analysis type, metric column, category column, and optional second metric. The LLM understands synonyms ("gender" → `SEX`, "emergency room spending" → `MONTHLY_ER_COST`) without hardcoded mappings. A simple keyword fallback runs if the LLM call fails.

**Step 3 — Compute.** Python runs the appropriate deterministic analytics functions. This happens twice — once for the primary question, and once for two supporting depth analyses chosen by rule.

**Step 4 — Narrate.** A single LLM call writes the report from the pre-computed JSON. The LLM is grounded with dataset facts so it cannot invent numbers.

For ML questions, the flow is different — see [section 4.2](#42-ml-workflow-autoresearch-loop).

---

## 4. The two workflows

### 4.1 Analytics workflow (2-pass engine)

This handles questions like "What is the distribution of cost?", "Is there a trend over time?", "Are male and female members significantly different in ER spend?"

**Pass 1 — Primary analysis**

The system identifies which of seven tools applies to the question:

| Analysis type | When it's used | Example question |
|---|---|---|
| `distribution_by_category` | metric sliced by a category | "distribution of cost by active status" |
| `trend_over_time` | metric across time periods | "how has cost trended monthly?" |
| `group_comparison` | is there a significant difference? | "compare ER cost between male and female" |
| `correlation_analysis` | relationship between two metrics | "correlate income and total cost" |
| `summary_stats` | descriptive statistics for a metric | "summarize monthly ER cost" |
| `entity_counts` | how many members / how many per group | "how many active members are there?" |
| `period_comparison` | entity-level diff between two time periods | "compare July vs September costs" |

The tool runs and returns a JSON object with statistics. No LLM is involved at this stage.

**Pass 2 — Depth analyses**

Two follow-up analyses are selected automatically based on what Pass 1 found. The rules are:

- If Pass 1 was a distribution → follow up with a statistical significance test, then a second dimension
- If Pass 1 was a comparison → follow up with the full distribution, then entity counts
- If Pass 1 was a trend → follow up with summary stats, then a distribution by segment
- If Pass 1 was a correlation → follow up with summary stats for each metric individually
- If Pass 1 was a period_comparison → follow up with the overall trend, then summary stats

This gives the final report more depth without asking the LLM to decide what to investigate.

**Narration**

All three results (primary + two depth) are bundled into a single prompt alongside the grain context. One LLM call writes the complete markdown report.

---

### 4.2 ML workflow (autoresearch loop)

This handles questions like "Build me a model to predict high-cost members."

The design is inspired by the autoresearch pattern: modify → run → measure → keep or discard → repeat. The key constraint is that the orchestrator controls the loop, not the agent. The agent only writes code; it never decides whether to keep or discard an experiment.

**Round 1**

The agent (no tools available) receives the data schema, a template pipeline, domain context, and grain facts. It writes a baseline classification pipeline as a Python code block. The orchestrator extracts the code with a regex, validates it (must write `metrics.json`, no network calls), then runs it via subprocess.

**Rounds 2 through 6**

The agent receives three things: the best pipeline code so far, the experiment log (every previous round's score, status, and description of the change), and the latest metrics. It makes exactly one targeted change — for example switching the model from Random Forest to LightGBM, or adding SHAP-based feature selection. The orchestrator runs the new pipeline, scores it, and decides: if the score improved, keep the new code; if not, revert to the previous best.

The composite score has five components:

| Component | Weight | What it measures |
|---|---|---|
| AUC | 45% | Discrimination ability |
| F1 | 20% | Precision-recall balance |
| Speed | 10% | Training time (penalizes slow models) |
| SHAP coverage | 10% | Top-10 feature explanation share |
| LLM explainability | 15% | How interpretable the model is to a non-technical audience |

**Stopping conditions**

The loop stops when any of these is true:
- Composite score reaches 0.65 or above
- No improvement in 2 consecutive rounds (plateau)
- Maximum of 6 rounds reached

**Final narration**

A single LLM call writes the final report from the experiment log, best metrics, and SHAP feature importances.

---

## 5. File-by-file guide

### `orchestrator.py` — Entry point and engine

This is where everything starts. It contains:

- `run_question()` — the top-level async function that routes and dispatches
- `_extract_params()` — LLM-based parameter extraction (question + column metadata → structured JSON); falls back to keyword matching if LLM fails
- `_run_primary()` — calls the correct analytics tool for Pass 1
- `_run_depth()` — rule-based selection and execution of the two depth analyses
- `_narrate()` — assembles the final LLM prompt and calls Claude
- `_run_ml_question()` — the outer autoresearch loop for ML questions
- `_run_ml_agent_round()` — a single round of agent code generation
- `_narrate_ml_results()` — writes the ML final report

### `agents/definitions.py` — Skill discovery and routing

- `discover_skills()` — scans the `registry/` folder and loads all manifests
- `route_question()` — scores the question against trigger phrases from each manifest; returns the best-matching skill
- `build_agent_prompt()` — assembles the full system prompt for an agent by combining `prompt.md`, `patterns.md`, `template.py`, and the company context files

### `tools/analytics.py` — Deterministic computation layer

Seven functions that do all the arithmetic. See [section 6](#6-the-analytics-tools-in-detail) for detail.

### `tools/data_tools.py` — Data loading, grain detection, MCP handlers

- `_load_df()` — loads CSV or parquet with a 5-minute in-memory cache
- `_detect_grain()` — figures out whether the data is panel (entity × time), cross-sectional (one row per entity), transactional, or time-series
- `build_grain_context()` — builds a human-readable paragraph summarizing the data: entity count, time periods, column list, aggregation guidance. This is injected into every agent prompt upfront.
- MCP handlers — thin wrappers that call analytics functions and format results for the agent

### `tools/ml_tools.py` — Pipeline execution and scoring

- `run_pipeline_handler()` — writes the agent's code to a temp file, runs it via subprocess (300-second timeout), and parses `metrics.json` and `shap_features.json` from the output
- `composite_score()` — computes the weighted score used for keep/discard decisions
- `validate_pipeline_code()` — checks syntax, confirms the code writes `metrics.json`, and blocks network imports

### `tools/servers.py` — MCP server wiring

Connects each tool name (as seen by the agent) to its handler function. There are two servers:

- **Data server** — 10 tools: profile_data, run_code, validate_cols, distribution, trend, comparison, correlation, summary, entity_counts, period_comparison
- **ML server** — 2 tools: run_pipeline, score_metrics

### `hooks/safety.py` — Safety guardrails

A `PreToolUse` hook that intercepts every tool call before execution. For Bash commands, it blocks patterns like `rm -rf`, `shutdown`, `reboot`, `kill -9`, `chmod 777`, and piped shell downloads (`curl | bash`, `wget | bash`). The sandboxed code executor in `data_tools.py` separately blocks Python imports of `os`, `sys`, `subprocess`, `open`, and `eval`.

### `registry/` — Skill definitions

Each subdirectory defines one skill:

```
registry/
  descriptive/
    manifest.json   — name, trigger phrases, tools needed, model
    prompt.md       — system prompt for the agent
    patterns.md     — code patterns and examples the agent can reference
  diagnostic/
    manifest.json
    prompt.md
    patterns.md
  classification/
    manifest.json
    prompt.md
    patterns.md
    template.py     — baseline pipeline template injected into Round 1
```

The manifest's `routing.trigger_phrases` list is what `route_question()` scores against. Longer, more specific phrases score higher than single-word triggers.

### `company_context/` — Domain knowledge

- `domain_knowledge.md` — healthcare cost analytics context: what the member archetypes are, what interventions exist, business definitions
- `metrics_definitions.md` — what each column means, its unit, and how to interpret it

These are appended to every agent system prompt so the LLM understands the business context without being told in the question.

### `Data/` — Input data

| File | Rows | Grain | Description |
|---|---|---|---|
| `pannel_data.csv` | 16,006 | Panel (member × month) | 5,478 members across Jul/Aug/Sep 2025 |
| `raw_data.csv` | 96,961 | Cross-sectional (one row per member) | 201 columns, used for ML classification |

### `output/` — Reports

Every run saves a timestamped markdown report: `output/YYYYMMDD_HHMMSS_skill_report.md`.

---

## 6. The analytics tools in detail

All seven functions follow the same internal steps:

**Step 1 — Grain detection.** The function calls `_detect_entity_time(df)` to identify which column is the entity identifier (e.g., `MEMBER_ID`) and which is the time column (e.g., `MONTH`). This determines whether the data is panel or cross-sectional.

**Step 2 — Panel collapse.** If the data is panel (multiple rows per entity), the function collapses it to one row per entity before computing statistics. This prevents the "averaging average" trap where you compute a mean of per-month averages instead of a true weighted mean.

**Step 3 — High-cardinality handling.** If a categorical column has more than 10 unique values, the system uses Pareto-aware bucketing: it keeps categories until they cover 95% of the metric's total value, OR the top 10 by entity/row count — whichever keeps more categories (capped at 10). This ensures high-value segments (e.g., a county with 50 members but 30% of total cost) are never buried in "Other." At least 3 categories are always kept.

**Step 4 — Statistics.** Each tool computes different things:

- `distribution_by_category` — for each group: mean, median, standard deviation, percentiles (p25, p50, p75, p90, p95, p99), count, share of total cost. Also computes a `comparison.direction` string stating which group is higher and by how much.
- `trend_over_time` — per-period mean and observation count, month-over-month percentage change, a linear regression (slope, R-squared, p-value), and an overall direction label (increasing / decreasing / stable).
- `group_comparison` — for two groups: Welch's t-test and Mann-Whitney U test with p-values, Cohen's d for effect size. For three or more groups: one-way ANOVA and Kruskal-Wallis with eta-squared.
- `correlation_analysis` — Pearson and Spearman correlations with p-values, and a strength label (strong / moderate / weak / negligible).
- `summary_stats` — mean, median, standard deviation, percentiles (p25 through p99), skewness, percentage of zero values, and a 95% confidence interval for the mean.
- `entity_counts` — unique entity count (using `nunique`, not row count), optionally broken down by category.
- `period_comparison` — entity-level diff between two time periods (panel data only). Returns six sections: **aggregate delta** (mean/sum/count change with pre-computed direction), **cohort matching** (matched/new/churned members with retention rate), **member movement** (% who increased/decreased/stayed flat, mean change per entity), **transition matrix** (cost bucket cross-tabulation showing who moved up/down), **top movers** (entities with biggest absolute changes), and **auto driver analysis** (see below).

**Auto driver analysis** (built into `period_comparison`)

Instead of requiring the user to specify which dimension to analyze, the system automatically loops through every categorical column in the dataset and computes per-segment contribution to the aggregate change. For each column:
1. High-cardinality columns (>10 values) are collapsed to top 10 + Other
2. Per-segment stats: mean change, total change, contribution %, % of members who increased/decreased
3. Segments with |contribution| < 5% are filtered out (noise reduction)
4. A flat `key_drivers` list ranks all impactful segments across all dimensions by absolute contribution

This surfaces answers like "Males drove 80.5% of the cost increase" and "Midland County contributed 29.3% of the delta" without the user needing to know which columns to look at.

**Step 5 — Output.** A JSON object with `analysis_type`, `metric`, `grain`, a `data_quality` section (null percentages per column), and the analysis-specific results.

---

## 7. How routing works

**Skill routing** (`route_question` in `agents/definitions.py`)

Each skill manifest lists trigger phrases. The question is scored against every phrase using token overlap. Multi-word phrases score higher because they are more specific. Ties are broken by the ratio of total score to phrase count. The skill with the highest final score wins. If no skill scores above zero, `descriptive` is used as the default.

Examples of trigger phrases by skill:
- `descriptive` — "distribution", "average", "how many", "over time", "trend", "count"
- `diagnostic` — "compare", "difference", "significant", "correlation", "relationship", "why"
- `classification` — "predict", "model", "classify", "build a model", "machine learning"

**Parameter extraction** (`_extract_params` in `orchestrator.py`)

Once a skill is selected, a single LLM call extracts the analysis parameters. The LLM receives:
- The user's question
- All column names, grouped by type (numeric vs categorical)
- Categorical column values (e.g., SEX has ["M", "F"])
- Entity and time column identifiers from grain detection

The LLM returns structured JSON: `{analysis_type, metric, category, metric_b, period_a, period_b, stratify_by}`. All returned column names are validated against the actual column list — any hallucinated column name is rejected and the default is used instead. For `period_comparison`, the LLM also receives the actual time period values from the data (e.g., `["2025-07-01", "2025-08-01", "2025-09-01"]`) and resolves user references like "July" or "September" to exact period values.

This replaces the previous keyword/fuzzy matching approach, which required hardcoded abbreviation dictionaries and couldn't handle synonyms. The LLM natively understands that "gender" maps to the `SEX` column, "emergency room spending" maps to `MONTHLY_ER_COST`, and "income" maps to `MEDIAN_HOUSEHOLD_INCOME_LATEST` — without any dataset-specific configuration.

If the LLM call fails, a simple keyword fallback runs (exact column name matching + basic analysis type keywords).

---

## 8. Safety guardrails

There are three layers of safety:

**Layer 1 — Bash hook** (`hooks/safety.py`). A `PreToolUse` callback runs before every agent tool call. If the tool is `bash` and the command matches any blocked pattern, the call is rejected immediately with a reason. Blocked patterns include: `rm -rf`, `rm -r`, `mkfs`, `:(){ :|:& };:` (fork bomb), `shutdown`, `reboot`, `kill -9`, `chmod 777`, `curl | bash`, `wget | bash`, `sudo`.

**Layer 2 — Code sandbox** (`tools/data_tools.py`). When the agent calls `run_code`, the Python string is executed via `exec()` in a restricted namespace. The namespace provides `df`, `pd`, `np`, and `scipy.stats` but nothing else. The code is scanned before execution and rejected if it contains: `import os`, `import sys`, `import subprocess`, `open(`, `__import__`, `exec(`, `eval(`.

**Layer 3 — Pipeline validation** (`tools/ml_tools.py`). Before running an ML pipeline, `validate_pipeline_code()` checks: that the code is syntactically valid Python, that it contains a `metrics.json` write statement (so the scoring system can read results), and that it does not import networking libraries (`socket`, `http`, `urllib`, `requests`).

---

## 9. Data layer

**Grain detection** (`_detect_grain` in `tools/data_tools.py`)

The system automatically determines the structure of any dataset:

- It scores every column as a potential entity identifier based on cardinality (medium-high unique count), data type (string or integer), and name patterns (columns named "ID", "MEMBER", "PATIENT", etc. score higher).
- It scores every column as a potential time column based on how many unique values it has (should be small for a time dimension), data type (date or string that parses as a date), and name patterns ("DATE", "MONTH", "PERIOD", etc.).
- It then tests whether the (entity, time) pair is unique per row. If so, it is panel data. If the entity alone is unique per row, it is cross-sectional. Other combinations produce "transaction" or "time-series" labels.

**Aggregation guidance**

The grain detection result includes aggregation guidance text — instructions like "to get per-entity metrics, use `df.groupby('MEMBER_ID')[metric].mean()`" and "to count entities, use `df['MEMBER_ID'].nunique()`." This is injected into agent prompts so agents compute correct aggregations automatically.

**Caching**

`_load_df()` caches the loaded DataFrame in memory with a 300-second TTL. Repeated calls with the same file path return the cached version without re-reading disk.

---

## 10. Cost and performance

**Analytics queries**

A typical analytics query costs approximately $0.05–$0.07 and completes in about 100 seconds. This includes two LLM calls: one for parameter extraction (~$0.005, ~5s) and one for narration (~$0.05, ~10s). The deterministic analytics tools run in milliseconds.

Before the current architecture, the system used a multi-agent loop where the LLM ran multiple rounds of computation. That approach cost $0.60–$1.14 per query and took 240–470 seconds, with numbers that varied between runs.

**ML queries**

An ML query runs 3–6 agent rounds plus pipeline execution. Each pipeline execution takes 60–120 seconds. Each agent code-generation call costs approximately $0.03–$0.05. Total is roughly $0.30–$0.60 and 5–10 minutes depending on how many rounds are needed.

---

## 11. Key design decisions and why

**Why does the LLM never compute numbers?**

LLMs are unreliable at arithmetic. In earlier versions of this system, the LLM was asked to compute averages, counts, and percentages. It frequently produced wrong numbers — not by much, but enough to make reports untrustworthy. Moving all computation to Python and giving the LLM only a role as a writer eliminated this class of error entirely.

**Why pre-compute the direction field?**

In testing, the LLM would sometimes say "active members have higher cost than inactive members" when the data showed the opposite. This happened because the LLM was inferring direction from numbers rather than being told explicitly. The `comparison.direction` string is computed by Python (which compares the group means directly) and the LLM is instructed to copy it verbatim.

**Why inject grain context upfront?**

Without grounding, the LLM would hallucinate context. It would say "over the past 12 months" when the data contained 3 months. It would reference column names that didn't exist. By building a data profile before any agent runs and injecting those facts into the system prompt, the LLM is constrained to what is actually in the data.

**Why does the orchestrator control the ML loop, not the agent?**

Agents are non-deterministic. If the agent controlled the loop, it might decide to run 10 experiments, or skip experiments, or change multiple things at once. By having the orchestrator manage keep/discard logic and round counting, experiments are reproducible and the stopping conditions are enforced.

**Why only one change per ML round?**

Changing multiple things at once makes it impossible to know which change improved the score. The single-change constraint produces a legible experiment log where each round has a clear description of what changed and whether it helped.

**Why use an LLM for parameter extraction instead of regex/fuzzy matching?**

The original implementation used hardcoded abbreviation dictionaries, substring matching, and token-overlap scoring to map user terms to column names. This was brittle: "gender" only worked because a dict mapped it to "SEX"; "emergency room spending" couldn't resolve to `MONTHLY_ER_COST`; any new dataset required manual shorthand updates. A single Sonnet call (~$0.005) replaces ~120 lines of brittle matching with native synonym understanding that generalizes to any dataset. Returned column names are validated against the actual schema, so LLM hallucination of non-existent columns is caught.

**Why a 2-pass design instead of an agent choosing follow-up analyses?**

Having the agent choose follow-up analyses introduces latency (an extra LLM call) and unpredictability (different follow-ups every run). The rule-based depth selection in Pass 2 is fast, consistent, and covers the analyses that are most useful for each analysis type.

---

## 12. Common usage examples

**Distribution question**
```bash
python3 orchestrator.py --data Data/pannel_data.csv \
  "What is the distribution of MONTHLY_TOTAL_COST by CURRENTLY_ACTIVE?"
```
Routes to: `descriptive` → `distribution_by_category` → depth: group_comparison + second dimension.

**Trend question**
```bash
python3 orchestrator.py --data Data/pannel_data.csv \
  "How has ER cost trended over the past few months?"
```
Routes to: `descriptive` → `trend_over_time` → depth: summary_stats + distribution by segment.

**Comparison question**
```bash
python3 orchestrator.py --data Data/pannel_data.csv \
  "Is there a significant difference in cost between male and female members?"
```
Routes to: `diagnostic` → `group_comparison` → depth: distribution + entity_counts.

**Count question**
```bash
python3 orchestrator.py --data Data/pannel_data.csv \
  "How many active members are there by county?"
```
Routes to: `descriptive` → `entity_counts` → depth: summary_stats + comparison.

**Period comparison question**
```bash
python3 orchestrator.py --data Data/pannel_data.csv \
  "Compare July vs September costs — what drove the change?"
```
Routes to: `diagnostic` → `period_comparison` → depth: trend + summary_stats. Auto driver analysis automatically decomposes the change across SEX, COUNTY, CITY, etc.

**ML question**
```bash
python3 orchestrator.py --data Data/raw_data.csv \
  "Build a model to predict TARGET_HIGH_COST_FLAG"
```
Routes to: `classification` → autoresearch loop → final narration.

**Quick mode (Pass 1 only, skip depth analyses)**
```bash
python3 orchestrator.py --data Data/pannel_data.csv --quick \
  "Summarize MONTHLY_ER_COST"
```

---

*Last updated: 2026-03-18*
