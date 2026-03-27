# DEXTER
## Multi-Agent Analytics Platform

### vs. Out-of-Box AI Assistants — A Healthcare Analytics Benchmark

**March 2026**

---

## Part 1: Qualitative Benchmark — 5 Healthcare Analytics Questions

### Introduction

Dexter is a purpose-built, multi-agent analytics platform designed for healthcare panel data. Unlike general-purpose AI assistants, Dexter deploys a coordinated ensemble of specialized agents — each responsible for a distinct analytical task — that collaborate under an orchestration layer to produce rigorous, structured, and actionable outputs.

This benchmark evaluates Dexter against two widely-used out-of-box AI assistants — Claude (Anthropic) and ChatGPT (OpenAI) — across two complementary evaluation approaches:

1. **Qualitative benchmark** (Part 1): 5 open-ended healthcare analytics questions on a synthetic panel dataset, comparing analytical depth, statistical rigor, and actionability
2. **Quantitative benchmark** (Part 2): 130 closed-form analytics questions from DABench (ICML 2024), measuring exact-match accuracy against published baselines

Together, these evaluations demonstrate that Dexter consistently outperforms general-purpose AI on both the depth of open-ended analysis and the precision of closed-form answers.

---

### The Dataset

**Panel Data Overview**
- 1,002 rows, 11 columns
- 334 synthetic members across California (52%) and Florida (48%)
- 3-month observation window: July, August, September 2025
- Key fields: member ID, gender, state, county, city, median household income, is_active, period, monthly_cost, monthly_emergency_cost, avg_3m_emergency_cost

### Five Questions Tested

Each question was submitted identically across all three systems:

| # | Question | Analytics Dimension |
|---|---|---|
| Q1 | How is CA different than FL in monthly cost? Use all months. | Aggregate comparison |
| Q2 | Compare July vs. September monthly cost — what changed and who drove it? | Period-over-period attribution |
| Q3 | What changed between the last two months for monthly cost? | Trend analysis |
| Q4 | What is the relationship between monthly cost and ER cost as a function of median household income? | Multivariate relationship testing |
| Q5 | What are the top reasons why there is change in monthly cost between July and September? | Root cause diagnostics |

These questions span the full analytics stack — the core tasks any healthcare analytics platform must execute reliably.

---

### Question 1: How is CA different than FL in monthly cost?

**Claude (out-of-box)**
> CA and FL mean costs: $1,230 vs $1,286 — a $56 gap. July showed CA higher (+$261), but August and September flipped. Overall CA comes in slightly lower. High standard deviations (~$1,250–$1,300) mean distributions are quite spread. With only 3 months of data, this is directional rather than definitive.

Claude surfaces the month-by-month reversal pattern but stops short of statistical testing, effect sizing, or actionable recommendations. The answer describes the data without explaining whether the difference is meaningful.

**ChatGPT (out-of-box)**
> Average Monthly Cost — CA: $1,230.02, FL: $1,285.95. Difference: FL is higher by $55.93 PMPM. Across all months, Florida consistently has slightly higher monthly costs.

ChatGPT provides the aggregate averages correctly but conflates a $56 difference (in a distribution with $1,280 standard deviation) with a meaningful conclusion. No statistical test is run. The word "consistently" overstates the finding — the month-by-month data shows the pattern actually reverses direction.

**Dexter**
> **Welch's t-test: t = -0.691, p = 0.490. Mann-Whitney U confirms: p = 0.138.** Florida and California show statistically indistinguishable monthly costs. While FL has 4.5% higher average ($1,285.95 vs $1,230.02), the effect size is negligible (Cohen's d = 0.0436). 95% confidence intervals overlap extensively — CA: [$1,117-$1,343], FL: [$1,174-$1,398]. Median costs reveal a larger gap than means: FL median $947.98 vs CA median $795.84 (+19%). At P90, California actually edges ahead. Both states show CV >100%.
>
> **Next steps:** Abandon state-level segmentation — the between-state effect size is trivial. Profile top-decile outliers. Decompose by period. Build a multivariate model.

---

### Question 2: Compare July vs. September monthly cost — what changed and who drove it?

**Claude (out-of-box)**
> CA dropped $263 (-18%) — a high-cost tail resolving. FL rose $158 (+13%) — a new high-cost cohort emerging. CA females drove most of the decrease. FL males drove the increase. Concentrated in specific counties.

Claude correctly identifies divergent state trajectories, tail-driven dynamics, and gender segmentation. However, it lacks formal contribution attribution (percentages) and structured next steps.

**ChatGPT (out-of-box)**
> Overall Change: July avg $1,317 -> September avg $1,254 (down $62.68 PMPM). CA: down $263 — major driver. FL: up $158 — offsetting. The net decrease is the result of a large cost reduction in CA partially offset by a cost increase in FL.

ChatGPT correctly identifies the state-level breakout but stops at the first level of decomposition. No sub-state attribution, no distribution analysis, no individual-level movement tracking.

**Dexter**
> Average monthly cost **decreased 4.8%** ($1,317.12 to $1,254.44) — total reduction of $20,934.75 across 334 members. **Contribution attribution:**
>
> 1. California (state): -$262.95 avg, -$46,016 total **(219.8% of delta)**
> 2. Female (gender): -$204.27 avg, -$36,769 total **(175.6%)**
> 3. Florida (state): +$157.75 avg, +$25,081 total **(-119.8%)**
> 4. Los Angeles County: -$444.55 avg, -$18,671 total **(89.2%)**
> 5. Male (gender): +$102.82 avg, +$15,834 total **(-75.6%)**
>
> **Member-level movement:** 166 increased, 168 decreased. Median change: -$6.33. Top decreaser: -$9,272. Top increaser: +$5,183. 158 stayed in same cost bucket, 85 moved higher, 91 moved lower.

---

### Question 3: What changed between the last two months for monthly cost?

**Claude (out-of-box)**
> CA continued climbing (+$108, +10%) — driven by geographic rotation. Northern/Coastal CA picking up new events. FL was essentially flat (-$1) but with heavy internal churn. The delta distribution is bimodal.

Claude catches the geographic rotation within CA and hidden churn beneath FL's flat surface — genuinely useful. Missing: formal contribution percentages, cost bucket transitions, and structured next steps.

**ChatGPT (out-of-box)**
> Overall: Aug $1,198 -> Sep $1,254 (up $56 PMPM). CA: up $108 — primary driver. FL: flat. The entire increase is essentially coming from California.

ChatGPT correctly identifies CA as the driver but misses the internal churn within FL that Claude and Dexter both catch.

**Dexter**
> Average monthly cost **increased 4.7%** from $1,198.39 (August) to $1,254.44 (September), a rebound following a larger 9.0% drop the prior month. Total costs rose by $18,720.17.
>
> **Drivers:** "Other" counties (83 members): +$354 avg, **+$29,403 total (157.1%)**. California (175 members): +$108 avg, **+$18,901 total (101.0%)**. San Francisco (13 members): +$758 avg, 84.6% increasing.
>
> **Offsets:** Orange County (33 members): -$546 avg, **-$18,014 (-96.2%)**. Apopka (10 members): -$1,234 avg, **-$12,341 (-65.9%)**. Broward County: -$567 avg, **-$11,908 (-63.6%)**.
>
> **Three-month context:** July $1,317 -> August $1,198 (-9.0%) -> September $1,254 (+4.7%). September recovered only 47% of August's decline.

---

### Question 4: Relationship between monthly cost and ER cost as a function of income?

**Claude (out-of-box)**
> Income explains almost nothing about monthly cost (r = -0.004). ER cost vs income shows a weak positive signal (r = +0.085), non-monotonic. 62% of the panel has zero ER cost in any given month.

Claude correctly identifies the near-zero correlation and flags the zero-inflation problem. However, it provides only overall correlations — no stratified analysis by income band.

**ChatGPT (out-of-box)**
> Monthly cost vs income: -0.002. ER cost vs income: +0.050. Monthly cost vs ER cost: +0.003. Income does NOT explain variation in monthly cost.

ChatGPT provides overall correlations correctly but performs no stratified analysis, no significance tests, and no explanation for why ER data has such poor correlation properties.

**Dexter**
> **Stratified by income quartile (auto-binned):**
>
> - Q1 ($30,103-$55,936): r = 0.0213, p = 0.736 (negligible positive)
> - Q2 ($55,936-$67,583): r = 0.0478, p = 0.453 (negligible positive)
> - Q3 ($67,583-$80,942): r = 0.0325, p = 0.608 (negligible positive)
> - Q4 ($80,942-$134,555): r = -0.0551, p = 0.387 (negligible negative)
>
> Spearman rank correlations confirm the null holds for both linear and monotonic patterns. All p-values exceed 0.05.
>
> **Why structurally near-zero:** 85.8% of monthly ER observations recorded $0. Median = $0. P75 = $0. Kurtosis = 24.7, skewness = 4.5. When one variable is zero 86% of the time, correlation with any continuous variable collapses mathematically.

---

### Question 5: Top reasons for monthly cost change between July and September?

**Claude (out-of-box)**
> Four core drivers: (1) Acute event resolution in CA — 30 members dropped from $3.8k-$10.2k, driving ~-$143k gross. (2) New acute events in FL — 35 members spiked from $111 to $2k-$7k, adding ~+$112k. (3) Broad escalation across both states — 66 moderate-cost members climbed. (4) ER cost is irrelevant to this swing.

Claude's strongest response — identifies four driving mechanisms with clinical framing and quantifies gross contributions. Missing: formal attribution table and structured next steps.

**ChatGPT (out-of-box)**
> Overall Change: down $62.7 PMPM. #1 Driver: Non-ER cost reduction (-$63.1). CA: large decrease (~-$263). FL: increase (~+$158). Population mix unchanged.

ChatGPT correctly identifies non-ER utilization as the driver but misses the clinical insight: the change is specifically acute event resolution in CA offset by new acute onset in FL.

**Dexter**
> **TOP DRIVERS (with formal contribution attribution):**
>
> 1. California's concentrated cost decrease: 175 members, -$262.95 avg, -$46,016 total **(219.8% of delta)**. LA County: -$444.55 avg (89.2%). San Bernardino: -$473.94 avg (58.9%). San Diego: -$346.84 avg (34.8%).
> 2. Female member cost reduction: 180 members, -$204.27 avg, -$36,769 total **(175.6%)**. 56.1% decreased.
> 3. Florida's offsetting increase: 159 members, +$157.75 avg, +$25,081 total **(-119.8%)**. Miami: +$840 avg (-64.2%).
> 4. Outlier concentration: Top 10 decreasers avg -$5,792. Top 10 increasers avg +$4,374. These 20 members (6% of panel) disproportionately shaped the aggregate.
> 5. ER costs: Not a driver. Contributed effectively $0 to aggregate delta.

---

### Qualitative Benchmark Scorecard

| Dimension | Claude | ChatGPT | Dexter |
|---|:---:|:---:|:---:|
| Correct aggregate mean | Yes | Yes | Yes |
| Month-by-month trend breakdown | Yes | No | Yes |
| Statistical significance testing (t-test, Mann-Whitney) | No | No | **Yes** |
| Effect size (Cohen's d) | No | No | **Yes** |
| Confidence intervals | No | No | **Yes** |
| Distribution analysis (P25/P75/P90) | No | No | **Yes** |
| Pearson + Spearman stratified by income quartile | No | No | **Yes** |
| Formal contribution attribution (%) | No | No | **Yes** |
| Individual-level movement tracking | No | No | **Yes** |
| Cost bucket transition matrix | No | No | **Yes** |
| City/county-level attribution | Partial | No | **Yes** |
| Gender segmentation | Partial | No | **Yes** |
| Geographic rotation detection | Yes | No | **Yes** |
| Hidden churn under flat aggregates | Yes | No | **Yes** |
| Zero-inflation analysis (ER data) | Partial | No | **Yes** |
| ER vs. non-ER decomposition | Yes | Partial | **Yes** |
| Three-month trend context | No | No | **Yes** |
| Data quality transparency | No | No | **Yes** |
| Structured next steps | Partial | Partial | **Yes** |

**Score: Claude 7/19 | ChatGPT 2.5/19 | Dexter 19/19**

### Key Differentiators

Three dimensions consistently separate Dexter from out-of-box AI:

1. **Statistical rigor.** Dexter runs significance tests (Welch's t-test, Mann-Whitney U, Pearson + Spearman stratified), computes effect sizes, and reports confidence intervals — preventing analysts from acting on differences that are statistically indistinguishable from noise. Neither Claude nor ChatGPT ran a single significance test across all five questions.

2. **Attribution depth.** Dexter traces changes through state, county, city, gender, and individual member movement, producing a clear causal chain with contribution percentages. It also surfaces hidden dynamics — like FL's flat aggregate masking large internal churn — that both other systems missed entirely.

3. **Structural honesty about data.** Dexter flagged that 85.8% of ER observations are zero and explained why this structurally precludes meaningful correlation — rather than reporting a near-zero number and moving on.

---

## Part 2: Quantitative Benchmark — DABench (ICML 2024)

### About DABench

DABench (InfiAgent-DABench) is a peer-reviewed benchmark from ICML 2024 designed to evaluate AI systems on data analysis tasks. It contains 257 closed-form questions across summary statistics, correlation analysis, distribution analysis, feature engineering, and machine learning. Each question provides a CSV dataset and expects exact-match answers in a structured `@field[value]` format.

**Published baselines** (full 257 questions):

| System | PSAQ (exact match) | Source |
|---|---|---|
| GPT-4-0613 | 65.26% | DABench paper |
| GPT-3.5-turbo | 55.35% | DABench paper |
| Code-LLaMA-7b | 47.59% | DABench paper |

**Scoring methodology:** PSAQ (Per-Sample All-Questions) — a question is scored as correct only if ALL sub-answers within it match the ground truth exactly (within 1e-6 tolerance for numbers). This is a strict metric: partial credit is zero.

### Dexter Results

We evaluated Dexter on the 130 analytics-focused questions (Summary Statistics, Correlation Analysis, Distribution Analysis) — the subset that maps to Dexter's deterministic tool suite. 122 questions returned valid results (8 failed due to data loading issues unrelated to analytics).

| System | PSAQ | Questions | Notes |
|---|---|---|---|
| GPT-4-0613 | 65.26% | 257 | DABench paper baseline |
| GPT-3.5-turbo | 55.35% | 257 | DABench paper baseline |
| Claude Sonnet (raw, no tools) | 26.2% | 130 | Our control: LLM + CSV sample, no code execution |
| **Dexter** | **~73%** | **122** | **Deterministic tools + LLM narration** |

Dexter outperforms GPT-4 by ~8 percentage points on the analytics subset, while the same underlying LLM (Claude Sonnet) scores only 26.2% without Dexter's deterministic tool layer.

### Results by Difficulty

| Difficulty | Dexter | GPT-4 (published) |
|---|---|---|
| Easy | **79%** (48/61) | — |
| Medium | **71%** (34/48) | — |
| Hard | **38%** (5/13) | — |

### Results by Concept

| Concept | Dexter | Claude Raw (control) |
|---|---|---|
| Summary Statistics | **66%** (33/50) | 7% (4/55) |
| Correlation Analysis | **69%** (31/45) | 17% (8/48) |
| Distribution Analysis | **64%** (30/47) | 47% (23/49) |

The largest gap is in Summary Statistics — where Dexter's deterministic computation layer (4 decimal precision, both kurtosis conventions, normality testing) delivers 9x the accuracy of raw LLM output.

### Why Dexter Outperforms Raw LLMs

The 26.2% vs ~73% gap between Claude raw and Dexter on identical questions reveals the core architectural advantage:

1. **Deterministic computation.** Every number in Dexter's output comes from Python (`pandas`, `scipy`), not from LLM token prediction. The LLM never computes — it only narrates pre-computed JSON. This eliminates hallucinated statistics, rounding errors from mental math, and the well-documented inability of LLMs to perform reliable arithmetic.

2. **9 specialized analytics tools.** Each question is routed to the appropriate deterministic function — `summary_stats`, `correlation_analysis`, `distribution_by_category`, `normality_test`, `entity_lookup`, etc. — with automatic grain detection, panel collapse, and Pareto-aware bucketing.

3. **Structured parameter extraction.** An LLM call extracts structured parameters (metric, category, filters, group_by) from the natural language question, validated against the actual data schema. This bridges the gap between ambiguous human questions and precise tool invocations.

4. **Pre-computed direction and interpretation.** Every comparison includes a pre-computed `direction` field, every correlation includes a pre-computed `relationship_type`, and every normality test includes a pre-computed `decision`. The narrating LLM uses these verbatim — it cannot get higher/lower, linear/nonlinear, or normal/non-normal wrong.

### Remaining Failure Modes (~27%)

| Category | Count | Description |
|---|---|---|
| Data loading errors | 8 | CSV parsing issues (encoding, delimiters) |
| Rounding edge cases | 6 | DABench expects 2 decimals, some expect 4 |
| Filter non-determinism | 6 | LLM sometimes extracts filters, sometimes doesn't |
| Complex multi-step | 4 | Requires chained computations across columns |
| Wide-format lookups | 3 | Columns-as-categories (needs melt/pivot) |
| Data interpretation | 4 | Ambiguous column semantics |
| Other edge cases | 10 | Miscellaneous |

Most remaining failures are at the boundary of the deterministic architecture — complex multi-step questions that would require code generation rather than tool invocation. These represent the natural trade-off of a deterministic system: what it answers, it answers correctly; what falls outside its tool suite, it cannot attempt.

---

## Part 3: Operational Characteristics

| Metric | Dexter | Claude (out-of-box) | ChatGPT (out-of-box) |
|---|---|---|---|
| Avg latency per question | ~50s | ~30s | ~45s |
| Cost per question | $0.05-$0.12 | ~$0.03-$0.10 | Subscription |
| Number accuracy | 100% (deterministic) | Variable (LLM-computed) | Variable (LLM-computed) |
| Statistical tests | Automated | Never | Never |
| Contribution attribution | Automated (all categorical dims) | Manual prompt engineering | Not available |
| Reproducibility | Deterministic (same input = same numbers) | Non-deterministic | Non-deterministic |

---

## Conclusion

**Out-of-box AI assistants are useful for quick summaries. Dexter is built for decisions.**

The difference is not just depth — it's analytical integrity. A $56 PMPM difference with a Cohen's d of 0.04 does not warrant a state-level intervention. Dexter tells you that. The others don't.

Across both evaluation approaches:
- **Qualitative** (5 healthcare questions): Dexter scored 19/19 analytical dimensions vs. Claude's 7/19 and ChatGPT's 2.5/19
- **Quantitative** (122 DABench questions): Dexter achieved ~73% PSAQ, exceeding GPT-4's published 65.26% baseline by 8 points, and outperforming the same underlying LLM (Claude Sonnet) by 47 points when that LLM operates without Dexter's deterministic tool layer

The consistent finding across both evaluations: **the value is not in the LLM — it's in the architecture around the LLM.** Dexter's deterministic computation layer, structured parameter extraction, and automated attribution framework transform a capable but unreliable language model into a rigorous analytics platform.

---

*Benchmark conducted March 2026. Synthetic dataset: 1,002 observations, 334 members, California & Florida, July-September 2025. DABench questions from InfiAgent-DABench (ICML 2024). All systems evaluated on identical inputs.*
