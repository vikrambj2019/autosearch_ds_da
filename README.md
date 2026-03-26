# Dexter

### The AI Data Scientist That Doesn't Stop at the First Answer

> Most tools answer your question.
> **Dexter tells you what you should have asked next.**

---

## The Problem

Every organization is drowning in data — but insight is still slow.

- Analysts are bottlenecks
- Dashboards answer predefined questions only
- AI tools generate fragile, unreliable analysis
- Decision-makers don't know what to ask next

> The gap isn't access to data. It's **depth of insight and speed of iteration.**

---

## Why Existing AI Falls Short

Even the most advanced commercial AI models struggle with real-world data analysis:

- Write inconsistent or non-reproducible code
- Misinterpret schemas and column mappings
- Perform incorrect statistical calculations
- Lose track of logic across multi-step analysis
- Hallucinate columns or metrics that don't exist
- Stop at surface-level answers

> These models are powerful for language — but **data analysis requires structure, determinism, and correctness.**

---

## The Solution

**Dexter is an AI-powered data scientist that:**

1. Takes a dataset + plain English question
2. Runs real statistical analysis (not hallucinated code)
3. Produces a clear, executive-ready report
4. Automatically generates deeper follow-up insights

```
Input: "What changed from last month?"

Output:
  Costs increased by 18%

  Drivers:
    • Chronic members  → +72% contribution
    • Alameda County   → +$2.1M impact
    • Top 5 members    → 40% of increase

  Follow-ups:
    • Gender-based breakdown
    • Statistical significance test
```

---

## What Makes Dexter Different

### 1. Deterministic + AI (Not Just AI)

Most tools: *"Write code → hope it works"*

Dexter: *"Use proven code → apply intelligence on top"*

- Pre-built statistical engine (Python) — 10 deterministic analytics tools
- AI only handles reasoning + narrative
- Every number computed in Python, never by the LLM
- **Zero fragile pipelines**

---

### 2. Always Goes One Level Deeper

Dexter doesn't stop at the answer.

Ask: *"Did costs increase?"*

Dexter returns:

- The answer (with statistical significance)
- Drivers of change (contribution attribution across all dimensions)
- Segment breakdown (state, county, city, gender — automatically)
- Key contributors (individual entity-level movers)

> This is the difference between **reporting** and **decision intelligence**.

---

### 3. Self-Improving Model Builder

Took a leaf out of [Andrej Karpathy's autoresearch repo](https://github.com/karpathy/autoresearch) and built a self-improving loop that actually works in practice — the orchestrator runs each experiment, scores it, and only keeps changes that move the needle.

Dexter builds predictive models like a real data scientist:

- Iterative improvement loop (up to 6 rounds)
- One controlled change per iteration
- Keeps only measurable improvements

The key insight: don't optimize for a single data science metric. Real-world models need to be good across the board — so Dexter scores each iteration on a **composite of AUC, interpretability, and inference speed**. A model that's 2% more accurate but twice as slow and impossible to explain isn't actually better.

---

### 4. Built for Real Data (Not Toy Datasets)

- Handles messy schemas
- Maps semantic meaning (e.g., `gender → SEX`)
- Detects panel vs cross-sectional vs transaction data
- Avoids common statistical errors (avg-of-avg, double-counting entities)
- Pareto-aware bucketing so high-value segments aren't buried in "Other"

---

## Architecture

```
User Question + CSV
        |
Data Profiler → Understands schema, grain, columns
        |
Parameter Extraction (LLM) → Maps natural language to exact tool params
        |
Deterministic Analysis Engine (Python, 10 tools)
        |
Depth Analysis (2 automatic follow-ups)
        |
AI Narrative Layer (Claude) → Writes report from pre-computed JSON
        |
Executive Report + Next Steps
```

### The 2-Pass Engine

- **Pass 1:** Deterministic primary analysis — Python computes all numbers. No LLM involved.
- **Pass 2:** Two rule-based depth analyses run automatically. Still no LLM.
- **Narrate:** A single LLM call turns all pre-computed JSON results into an executive report. The LLM cannot get numbers wrong because it never computes them — it only narrates.

### The ML Workflow

For classification questions, Dexter uses an autoresearch-inspired experiment loop:

1. Agent profiles data + generates baseline pipeline code
2. Orchestrator runs the pipeline, measures composite score
3. Agent makes ONE targeted change per round
4. Orchestrator runs, compares: KEEP (if better) or DISCARD (if worse)
5. Repeat until convergence (up to 6 rounds)

The orchestrator controls the loop — the agent generates code but does NOT execute it.

---

## Analytics Tools

10 deterministic functions in `tools/analytics.py`:

| Tool | What It Does |
|---|---|
| `distribution_by_category` | Distribution/breakdown of metric by category |
| `trend_over_time` | Time series trend with regression stats |
| `group_comparison` | Statistical test between groups (t-test, Mann-Whitney, Cohen's d) |
| `correlation_analysis` | Pearson/Spearman correlation + relationship type |
| `summary_stats` | Percentiles, skewness, kurtosis, 4-decimal precision |
| `normality_test` | Shapiro-Wilk, KS, Anderson-Darling with pre-computed decision |
| `entity_lookup` | "Which X has highest/lowest Y?" with ranked leaderboard |
| `entity_counts` | Unique entity counts, optionally grouped |
| `price_volume_mix` | Revenue/margin decomposition: price, volume, mix, new, lost effects |
| `period_comparison` | Entity-level diff between periods + auto driver attribution |

Every tool is grain-aware (panel, cross-sectional, transaction), handles `filters` and `group_by`, and pre-computes direction so the LLM can't get higher/lower wrong.

---

## Benchmark Results

### DABench (ICML 2024) — Closed-Form Analytics

| System | PSAQ (Exact Match) | Notes |
|---|---|---|
| GPT-3.5-turbo | 55.35% | DABench paper |
| GPT-4-0613 | 65.26% | DABench paper |
| Claude Sonnet (raw, no tools) | 26.2% | Same LLM, no Dexter |
| **Dexter** | **~73%** | 122 analytics questions |

Dexter outperforms GPT-4 by ~8 points. The same underlying LLM (Claude Sonnet) scores only 26.2% without Dexter's deterministic tool layer — **proving the value is in the architecture, not the model.**

### Qualitative Benchmark — 5 Healthcare Analytics Questions

Side-by-side comparison against Claude and ChatGPT on open-ended healthcare analytics:

| Dimension | Claude | ChatGPT | Dexter |
|---|---|---|---|
| Statistical significance testing | No | No | Yes |
| Effect size (Cohen's d) | No | No | Yes |
| Confidence intervals | No | No | Yes |
| Formal contribution attribution (%) | No | No | Yes |
| Individual-level movement tracking | No | No | Yes |
| Cost bucket transition matrix | No | No | Yes |
| City/county-level attribution | Partial | No | Yes |
| Income-stratified correlation | No | No | Yes |
| Three-month trend context | No | No | Yes |

**Score: Claude 7/19 | ChatGPT 2.5/19 | Dexter 19/19**

Full benchmark: [`dexter_benchmark_combined.md`](dexter_benchmark_combined.md)

---

## Quick Start

```bash
pip install anthropic pandas numpy scipy scikit-learn lightgbm xgboost shap tabulate

# Descriptive
python3 orchestrator.py --data Data/pannel_data.csv "What is the distribution of MONTHLY_TOTAL_COST by CURRENTLY_ACTIVE?"

# Diagnostic
python3 orchestrator.py --data Data/pannel_data.csv "Is there a significant difference in cost between male and female?"

# Period comparison
python3 orchestrator.py --data Data/pannel_data.csv "Compare July vs September costs"

# Price-Volume-Mix
python3 orchestrator.py --data Data/transaction_data_v2.csv "Why did revenue change? Price volume mix analysis"

# ML model build
python3 orchestrator.py --data Data/raw_data.csv "Build me a classification model predicting TARGET_HIGH_COST_FLAG"

# Quick mode (Pass 1 only)
python3 orchestrator.py --data Data/pannel_data.csv --quick "How many members are there?"
```

### Sample Data

| Dataset | Description |
|---|---|
| `Data/pannel_data.csv` | Panel — 1,002 rows (334 members × 3 months) |
| `Data/raw_data.csv` | Cross-sectional — 1,000 rows, 201 columns |
| `Data/transaction_data_v2.csv` | Transaction — 73 rows (9 customers × 8 SKUs × 2 months) |

---

## Trust & Reliability

- All numbers computed in Python (not AI)
- AI cannot alter results — it only narrates pre-computed JSON
- Local-first: data never leaves your environment
- Full audit trail with timestamps
- Deterministic: same input = same numbers
- Built for enterprise trust from day one

---

*Built by Vikram Bandugula — March 2026*
