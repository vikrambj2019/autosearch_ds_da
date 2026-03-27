# Analytics Report

**Question:** What changed between 2025 and 2026? Perform price volume mix analysis
**Data:** Data/pvm_analysis.csv
**Skill:** diagnostic
**Timestamp:** 2026-03-26 17:12:02
**Turns:** 1 | **Duration:** 35445ms

---
# Price-Volume-Mix Analysis: 2025 vs. 2026
**Dataset:** pvm_analysis.csv | **Grain:** Transaction-level, aggregated to product × customer pairs | **Method:** Laspeyres decomposition (base-period weights) | **Pricing Grain:** GFS Pcode Desc × Sold To Number | **Matched Pairs:** 18

---

## Executive Summary

Revenue **increased** by **$10,133** (+6.32%) from **$160,402** in 2025 to **$170,535** in 2026. This growth was driven entirely by volume, which contributed **$13,106.41** (+129.3% of total change). Both price and mix acted as headwinds, partially offsetting volume gains with a combined drag of **−$2,973.41**. There were no new or lost customer-product pairs between the two periods.

---

## Key Finding: Volume Carried the Growth; Price and Mix Eroded It

The PVM decomposition tells a clear story:

| Effect | $ Impact | % of Total Change |
|---|---|---|
| Volume | +$13,106.41 | +129.3% |
| Price | −$1,640.49 | −16.2% |
| Mix | −$1,332.92 | −13.2% |
| New/Lost | $0.00 | 0.0% |
| **Total** | **+$10,133** | **100%** |

Volume growth of **4,110 units** (+8.17%, from **50,300** to **54,410**) was the sole engine of revenue growth. However, average unit price slipped from **$3.1889** to **$3.1343**, compressing margin across the portfolio.

---

## Deeper Insights

### Top Contributor: Steno Diabetes Center — 3 ml Reservoir
The single largest revenue swing came from **3 ml Paradigm Rsvr | Steno Diabetes Center A/S**, which added **$11,840** in incremental revenue. Volume surged from **5,910 to 9,740 units**, generating a volume effect of **+$1,579.04** and a mix effect of **+$10,675.78**. Price softened slightly (−2.15%, from $3.2699 to $3.1997), costing **−$414.82**, but the sheer volume and favorable mix more than compensated.

### Largest Drag: ASDIA Est. Strasbourg — 1.8 ml Reservoir
The biggest detractor was **1.8 ml Paradigm Rsvr | ASDIA Est. Strasbourg**, down **−$7,413**. Volume fell from **11,480 to 9,360 units**, and the mix effect alone was **−$9,490.97**. Price also declined 2.44% (from $3.157 to $3.08), adding a price drag of **−$883.35**. A large, shrinking SKU with negative mix is the most concerning dynamic in the portfolio.

### Notable Price Outliers
- **HD MEDICAL — 1.8 ml Reservoir** saw price spike +145.63% (from $2.915 to $7.16), but volume collapsed from 400 to just 50 units, resulting in a net revenue decline of **−$808**.
- **Airmedic — 3 ml Reservoir** showed a dramatic price increase of +550.0% ($0.60 to $3.90), though off a very small base (revenue change of only +$72).
- **HD MEDICAL — 3 ml Reservoir** experienced a −33.0% price decline, contributing **−$256.75** in price effect, and volume fell from 150 to 40 units.

### Portfolio Stability
All 18 product-customer pairs present in 2025 were still active in 2026 (0 new, 0 lost pairs), indicating strong customer retention. The distribution of sales is highly skewed (skewness: 2.8652), with the top two pairs — ASDIA 3 ml ($82,439 current) and ASDIA 1.8 ml ($28,829 current) — dominating total revenue.

---

## Data Notes

- The dataset contains **36 rows across 34 unique Sales entities** across 2 periods (2025, 2026). The panel is **unbalanced**.
- **GFS Std Level 8 Desc** is a constant column (only 1 unique value) and carries no analytical signal.
- The Sales distribution is right-skewed (skewness: 2.8652, mean-to-median ratio: 11.9308), indicating a small number of high-value pairs dominate the totals. Results should be interpreted with this concentration in mind.
- No null values were detected in the Sales column (0.0% null rate).

---

## Next Steps

1. **Investigate ASDIA 1.8 ml volume decline** — this pair lost 2,120 units and generated the largest negative mix effect (−$9,490.97). Understand whether this reflects a product substitution, a lost order, or a deliberate portfolio shift.
2. **Review HD MEDICAL pricing** — the 145.63% price increase on 1.8 ml drove customers away (400 → 50 units). Assess whether this was intentional repricing or a data entry issue.
3. **Protect the Steno Diabetes Center relationship** — this pair delivered +$11,840, the largest single gain. Ensuring contract continuity here should be a retention priority.
4. **Monitor average unit price erosion** — the portfolio-wide price declined from $3.1889 to $3.1343. With volume as the sole growth driver, any volume softness in 2027 would directly threaten revenue.