# Analytics Report

**Question:** What are the top reasons why there is a change in monthly cost between July and September?
**Data:** Data/pannel_data.csv
**Skill:** diagnostic
**Timestamp:** 2026-03-26 16:47:41
**Turns:** 1 | **Duration:** 35344ms

---
# Monthly Total Cost Change: July to September 2025
### Analytics Report — Panel Data, 334 Members, 3 Monthly Periods

---

## Executive Summary

Average MONTHLY_TOTAL_COST decreased by 5.8% from $1,267.51 to $1,194.13 between July and September 2025, representing a total aggregate reduction of $24,509.16 across 334 members. The decline was not uniform: costs first rose 3.0% in August (mean: $1,306.05) before falling 8.6% in September, suggesting the July-to-September net decrease was driven primarily by a sharp September reversal. The top drivers of this decrease are concentrated in male members and specific Michigan counties and cities.

**Data grain:** Panel dataset (1,002 rows × 11 columns), one observation per member per month, 334 unique members matched across all three periods. The panel is fully balanced with 100.0% retention — no new or churned entities.

---

## Key Finding

**Male members and Oakland County account for the largest shares of the overall cost decrease.**

Male members (154 entities) drove 81.1% of the total cost reduction, with a mean change of -$129.13 and a total change of -$19,886.00. Female members (180 entities) contributed the remaining 18.9%, with a more modest mean change of -$25.68 and a total change of -$4,623.16.

Geographically, Oakland County contributed 59.0% of the total delta (-$14,452.79 across 54 members, mean change -$267.64), while Genesee County contributed 55.0% (-$13,481.76 across 21 members, mean change -$641.99) — the sharpest per-member decline of any county analyzed.

---

## Deeper Insights

### Member-Level Movement
Of 334 matched members, 160 (47.9%) saw costs increase and 174 (52.1%) saw costs decrease — none were flat. The mean change per entity was -$73.38, but the median change was a more modest -$42.80, and the standard deviation of individual changes was $1,761.11, indicating high dispersion. A small number of extreme movers are materially shaping the aggregate result.

**Top individual increases:**
- Member `41184a66` surged by +$9,251.75
- Member `9cb9c7ab` increased by +$5,648.13

**Top individual decreases:**
- Member `fce80292` fell by -$5,622.32
- Member `9b906aa6` fell by -$5,478.01

### Cost Bucket Transitions
Using thresholds at the p50 ($883.92) and p90 ($2,826.82), 152 members stayed within the same cost bucket, 81 moved to a higher cost bucket, and 101 moved to a lower cost bucket — consistent with the overall decreasing direction.

### Geographic Drivers
Beyond Oakland and Genesee, **Wayne County** (90 members) contributed 40.5% of the total delta (-$9,925.53, mean change -$110.28). Notably, **Saginaw County** (17 members) moved in the opposite direction, contributing -38.7% to the delta (i.e., costs *increased* there by a total of +$9,479.50, mean change +$557.62 per member, with 70.6% of members increasing).

At the city level, **Livonia** (24 members) and **Clinton Township** (15 members) were the top city-level contributors to decreases, with total changes of -$11,366.84 (mean -$473.62; 75.0% of members decreased) and -$10,719.67 (mean -$714.64; 60.0% decreased), respectively. **Sterling Heights** (14 members) and **Westland** (19 members) bucked the trend, posting total increases of +$4,682.73 and +$5,429.79, respectively.

### Distribution Context
Across all 1,002 observations, MONTHLY_TOTAL_COST is right-skewed (skewness: 2.1534), with a mean-to-median ratio of 1.4035 — confirming that high-cost outliers pull the average up. The p99 value is $5,661.54 and the maximum is $10,198.54, consistent with the extreme individual movers noted above.

---

## Data Notes

- **STATE_LATEST** is a constant column (only 1 unique value) and was excluded from driver analysis.
- **CITY_LATEST** had 26 categories and was collapsed for analysis; 9 city segments are reported.
- MONTHLY_TOTAL_COST has 0 null values across all 1,002 rows.
- The three-point regression (slope: -36.69 per period, R²: 0.4164, p-value: 0.5535) should be interpreted with caution given only 3 data points.

---

## Next Steps

1. **Investigate male member cost drivers specifically** — with 81.1% of the total delta attributable to male members, a deeper clinical or utilization review of this cohort is warranted.
2. **Audit Genesee County members** — 21 members averaged a -$641.99 change, the steepest per-member decline of any county; validate whether this reflects genuine utilization reduction or data anomalies.
3. **Examine Saginaw County's counter-trend** — 70.6% of its 17 members saw costs increase (+$9,479.50 total), running opposite to the overall population trend.
4. **Investigate the August spike** — costs rose 3.0% in August before falling 8.6% in September; understanding what drove that intermediate increase may clarify whether September's drop is a correction or a new trend.
5. **Profile the top individual movers** — the member with a +$9,251.75 increase and those with decreases exceeding -$5,000 are disproportionately influencing aggregate results and merit case-level review.