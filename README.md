# Dexter — AI Analytics Assistant for Any  Data

Ask a question in plain English. Get a full analytics report in seconds.

Dexter is an AI-powered analytics system built for various analysis. It connects directly to your data, runs rigorous statistical analysis, and writes a clear report — no dashboards to navigate, no SQL to write, no analyst queue to join.

---

## What It Does

You type a question. Dexter figures out what analysis to run, runs it correctly on your data, and hands you back a written report with tables, statistics, and a clear narrative.

**Examples of questions you can ask:**

| Question | What Dexter does |
|---|---|
| "What is the distribution of monthly cost by active status?" | Breaks down cost across member groups with statistical significance testing |
| "Is there a significant difference in cost between male and female members?" | Runs t-test, Mann-Whitney U, and Cohen's d effect size |
| "How have costs trended from July to September?" | Time series analysis with regression stats |
| "Compare July vs September costs — who drove the change?" | Entity-level diff, auto driver analysis across all categories, top movers |
| "Build me a model predicting high-cost members" | Trains and iteratively improves a classification model, reports AUC and key drivers |

---

## How It Works

Dexter uses a two-step approach designed so the AI never makes up numbers:

1. **Python computes everything.** All statistics, comparisons, and model scores are calculated by deterministic Python code — not by the AI.
2. **The AI only writes the narrative.** Once the numbers are ready, the AI reads the pre-computed results and writes the English report. It cannot get the numbers wrong because it doesn't produce them.

For model building, Dexter uses a **self-improving autoresearch loop** — inspired by [Andrej Karpathy's autoresearch](https://github.com/karpathy/autoresearch).

Here is how it works:

- **Round 1** — The AI reads your data schema and writes a baseline classification model pipeline from scratch.
- **Rounds 2–6** — The AI sees the best pipeline so far plus a log of every previous experiment. It makes exactly **one targeted change** — for example, switching from Random Forest to LightGBM, or adding feature selection. Dexter runs the new pipeline, scores it, and decides: keep the change if it improved, discard it if it did not.
- **Stops automatically** when the model hits a quality threshold, stops improving for two rounds in a row, or reaches 6 rounds.

Each model is scored on five dimensions: prediction accuracy (AUC), precision-recall balance (F1), training speed, how well the top features explain the model (SHAP coverage), and how interpretable the results are to a non-technical audience.

The AI proposes changes but **never decides what to keep** — that is always Dexter's job. This means experiments are consistent, reproducible, and the log clearly shows what each round changed and whether it helped.

---

## What It Knows About Your Data

Dexter is pre-loaded with context about healthcare cost analytics:

- How to handle **panel data** (members observed across multiple months) without averaging incorrectly
- That cost distributions are **right-skewed** — median is more meaningful than mean for most questions
- That **active vs inactive members** have different profiles and should be segmented
- That **top 5% of members often account for 40–60% of total spend**
- That ER costs are episodic and county-level variation reflects network differences, not health status

---

## Sample Output

> *"Across 334 unique members observed over 3 monthly periods (July–September 2025), average monthly healthcare costs are nearly identical between active and inactive members. The gap is statistically negligible — enrollment status alone does not meaningfully differentiate cost levels in this panel.*
>
> *Active members account for 92.3% of total spend ($1.16M of $1.26M overall), simply by virtue of representing 308 of the 334 members. While the means are close, the inactive group has a notably higher median ($958 vs $883), suggesting the active group has a longer right tail. Two independent statistical tests confirm no significant difference: Welch's t-test p = 0.89, Mann-Whitney p = 0.85, Cohen's d = 0.016 (negligible effect)."*

Reports include tables, statistical test results, effect sizes, and follow-up analyses — all generated automatically.

---

## Types of Analysis

| Type | When to use |
|---|---|
| **Descriptive** | Distributions, breakdowns, counts, trends over time |
| **Diagnostic** | Group comparisons, significance testing, correlation |
| **Period Comparison** | Month-over-month or period-over-period change with driver analysis |
| **Predictive (ML)** | Classification models to identify high-cost or high-risk members |

---

## Data It Works With

- **Panel data** — members observed across multiple time periods (e.g. monthly snapshots)
- **Cross-sectional data** — one row per member
- Input format: CSV files

Dexter auto-detects which type of data you've provided and adjusts all analysis accordingly.

---

## Guardrails

- The AI never executes arbitrary code on your system — all tool calls go through a safety layer
- The AI never computes numbers — only Python does
- All reports are saved with timestamps to the `output/` folder for auditability

---

## Built With

- [Claude Agent SDK](https://www.anthropic.com) — AI reasoning and report writing
- Python — all statistical computation (pandas, scipy, scikit-learn, LightGBM, XGBoost)
- Runs entirely locally — your data never leaves your machine
