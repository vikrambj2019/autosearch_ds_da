# Analytics Report

**Question:** What is the distribution of MONTHLY_TOTAL_COST by CURRENTLY_ACTIVE?
**Data:** Data/pannel_data.csv
**Skill:** descriptive
**Timestamp:** 2026-03-21 10:22:20
**Turns:** 1 | **Duration:** 29770ms

---
# Analytics Report: MONTHLY_TOTAL_COST Distribution by CURRENTLY_ACTIVE

---

## Executive Summary

Across 334 unique members observed over 3 monthly periods (July–September 2025), average monthly healthcare costs are nearly identical between active and inactive members. The gap is statistically negligible, meaning enrollment status alone does not meaningfully differentiate cost levels in this panel.

**Grain & Aggregation:** Panel data — 334 entities × 3 time periods = 1,002 observations. Means are computed as total_sum / obs_count (not averages of averages). Entity counts use nunique.

---

## Key Finding

**True has 1.6% higher average MONTHLY_TOTAL_COST ($1,257.44 vs $1,237.57).**

This near-identical average cost across active (CURRENTLY_ACTIVE = True) and inactive (CURRENTLY_ACTIVE = False) members is the headline result. The difference of roughly $20 per member per month is not statistically significant by any test applied.

---

## Deeper Insights

### Group Size and Cost Share Imbalance
The two groups are highly unequal in size:

| Group | Entities | Observations | Mean Cost | Median Cost | Total Cost | Cost Share |
|-------|----------|--------------|-----------|-------------|------------|------------|
| True (Active) | 308 | 924 | $1,257.44 | $883.28 | $1,161,875.28 | 92.3% |
| False (Inactive) | 26 | 78 | $1,237.57 | $958.06 | $96,530.24 | 7.7% |

Active members account for 92.3% of total spend ($1,161,875.28 of $1,258,405.52 overall), simply by virtue of representing 308 of the 334 members in the panel.

### Distribution Shape Differs Despite Similar Means
While the means are close, the inactive group has a notably higher median ($958.06 vs $883.28), suggesting the active group has a longer right tail pulling its mean upward. This is confirmed by the active group's higher maximum ($10,198.54 vs $5,972.68) and higher 90th percentile ($2,852.55 vs $2,699.00). The inactive group's standard deviation is also lower ($1,191.64 vs $1,270.12), indicating a tighter cost distribution.

### Statistical Tests: No Significant Difference
Two independent statistical tests confirm the groups are indistinguishable on cost:

- **Welch's t-test:** t = -0.1407, p = 0.888413 — not significant at the 0.05 level
- **Mann-Whitney U test:** U = 36,492.0, p = 0.852769 — not significant at the 0.05 level
- **Cohen's d effect size:** 0.0157 — interpreted as **negligible**

The 95% confidence intervals further illustrate this: active members CI = ($1,175.44–$1,339.44); inactive members CI = ($968.89–$1,506.24). The wide overlap leaves no meaningful separation between groups.

### Secondary Finding: Sex Drives More Cost Variation Than Active Status
A secondary breakdown by SEX reveals a more meaningful gap: **M has 9.8% higher average MONTHLY_TOTAL_COST ($1,319.40 vs $1,201.56)**. Male members (154 entities, 462 observations) average $1,319.40/month vs. $1,201.56 for female members (180 entities, 540 observations). Males account for 48.4% of total spend ($609,564.66) and females 51.6% ($648,840.86). The 9.8% sex-based gap is six times larger than the 1.6% active-status gap, suggesting sex is a more informative cost predictor in this panel.

---

## Data Notes

- **No missing values:** MONTHLY_TOTAL_COST has 0 nulls across all 1,002 rows (0.0% null rate).
- **Zero costs:** Neither the active nor inactive group recorded any zero-cost observations (zero_pct = 0.0 for both).
- **STATE_LATEST** is a constant column (only 1 unique value) and cannot be used for geographic segmentation.
- **Panel is balanced:** All 334 entities appear across all 3 time periods.
- The CURRENTLY_ACTIVE column does not appear in the dataset's original column list — its source and definition should be confirmed prior to any operational use.

---

## Next Steps

1. **Investigate cost drivers within the active cohort** — with 308 entities and a wide range ($0.55–$10,198.54), segmenting active members by cost tier or utilization pattern would be more actionable than the active/inactive split.
2. **Explore the sex-based cost gap** — the 9.8% difference between male and female members warrants deeper analysis, including ER cost breakdowns using MONTHLY_ER_COST and AVG_3M_ER_COST.
3. **Clarify CURRENTLY_ACTIVE definition** — if this field changes over the panel window, a member-level trajectory analysis may reveal cost trends around transitions in status.
4. **Extend observation window** — with only 3 time periods, trend confidence is limited; additional months would improve statistical power.