# Analytics Report

**Question:** What changed between 2025 and 2026? Perform price volume mix analysis
**Data:** Data/pvm_analysis.csv
**Skill:** diagnostic
**Timestamp:** 2026-03-26 17:10:56
**Turns:** 1 | **Duration:** 27007ms

---
# Price Volume Mix Analysis: 2025 vs. 2026

**Dataset:** Data/pvm_analysis.csv | **Grain:** Transaction-level, aggregated to entity totals by Sales entity | **Periods:** 2025 and 2026

---

## Executive Summary

A price-volume-mix (PVM) decomposition was attempted across 2 time periods (2025 and 2026) covering 34 unique Sales entities across 36 observations. Unfortunately, the automated PVM engine was unable to complete the year-over-year decomposition due to a technical detection issue with the time column. The directional split (price effect, volume effect, mix effect) could not be computed from the available results. What can be reported is a robust statistical profile of the Sales metric across the full dataset, which provides meaningful context for the scale and distribution of revenue under analysis.

---

## Key Finding

**The PVM engine returned an error:** *"No time/date column detected. Pass time_col explicitly."*

As a result, no computed price effect, volume effect, or mix effect figures are available from this run. The 2025-to-2026 directional changes cannot be stated without risking invention of numbers not present in the JSON results. This must be resolved before a formal PVM narrative can be delivered.

---

## Deeper Insights: Sales Distribution Profile

Despite the PVM failure, the summary statistics across all 36 observations offer important context about the shape of the business:

- **Total Sales (all periods combined):** $330,937.00
- **Number of Sales entities:** 34, observed across 36 rows (indicating 2 entities appear in both periods, or a small number of split transactions)
- **Mean Sales per observation:** $9,192.69
- **Median Sales:** $770.50
- **Mean-to-median ratio:** 11.93 — a strong signal that a small number of large transactions dominate the revenue base

**The distribution is highly right-skewed and leptokurtic:**

- Skewness: 2.8652 (heavily right-tailed; a few large deals pull the average far above the median)
- Kurtosis (excess): 8.1409 (extreme outlier concentration relative to a normal distribution)
- 32 out of 36 values fall within 1 standard deviation of the mean ($9,192.69 ± $19,717.78)
- All 34 entities fall within 2 standard deviations

**The revenue range is extremely wide:**

| Percentile | Sales Value |
|---|---|
| Minimum | $6.00 |
| 25th percentile | $284.00 |
| Median | $770.50 |
| 75th percentile | $5,975.25 |
| 90th percentile | $29,997.00 |
| 95th percentile | $46,407.75 |
| 99th percentile | $80,502.10 |
| Maximum | $82,439.00 |

The bottom quartile of transactions contributes negligibly (capped at $284), while the top 10% of observations exceed $29,997 each — a classic long-tail revenue pattern that will make PVM interpretation sensitive to mix effects once the decomposition is re-run.

**95% Confidence Interval for mean Sales:** $2,521.16 to $15,864.23, reflecting the high variance in transaction sizes.

---

## Data Notes

1. **PVM engine failure:** The `Year` column was not auto-detected as a time column. The column must be passed explicitly as `time_col='Year'` to enable the decomposition.
2. **Constant column warning:** `GFS Std Level 8 Desc` contains only 1 unique value across all rows and should be excluded from segmentation analysis — it will contribute nothing to mix analysis.
3. **Unbalanced panel:** The dataset is flagged as an unbalanced panel (34 entities, 36 rows). This means not all entities appear in both 2025 and 2026, which will affect the volume and mix calculations once PVM is re-run — entities appearing in only one period need to be handled explicitly (e.g., as entries or exits).
4. **No null values** were detected in the Sales column (0.0% null rate across 36 rows).

---

## Next Steps

1. **Re-run the PVM engine** passing `time_col='Year'` explicitly to resolve the detection error and generate the price, volume, and mix decomposition.
2. **Decide on panel handling:** Determine whether the 2 entities missing from one period represent true exits/entries or data gaps — this choice will materially affect the mix effect calculation.
3. **Segment by Country and GFS Pcode Desc** once PVM results are available, given the extreme skew — the top-decile entities likely drive the bulk of any year-over-year change.
4. **Drop `GFS Std Level 8 Desc`** from all segmentation cuts given its constant value.