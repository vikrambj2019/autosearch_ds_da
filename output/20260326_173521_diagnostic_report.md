# Analytics Report

**Question:** What changed between 2012 and 2013? Perform price volume mix analysis
**Data:** Data/pvm_v2.csv
**Skill:** diagnostic
**Timestamp:** 2026-03-26 17:35:21
**Turns:** 1 | **Duration:** 41823ms

---
# Price Volume Mix Analysis: 2012 vs. 2013

---

## Executive Summary

Revenue **increased** by **$195,858.79 (38.4%)** from **$509,986.14** in 2012 to **$705,844.93** in 2013. However, this headline figure masks a structurally complex story: the growth was driven almost entirely by new product-customer relationships entering the portfolio, while the continuing book of business showed volume contraction and minimal pricing power. Customer churn (lost pairs) consumed the equivalent of **-220.8%** of base revenue, making new business acquisition the critical lifeline of 2013 performance.

**Grain & Aggregation:** Analysis is at the transaction level (3,000 rows, 2,738 unique Qty Discount entities). PVM decomposition is performed at the **Product × Customer** pricing grain using the **Laspeyres (base-period weights)** method, with revenue aggregated by summing transactions within each year.

---

## Key Finding: New Business Drove Everything

Of the **$195,858.79** total revenue change, the four PVM components tell a stark story:

| Effect | $ Impact | % of Base Revenue |
|---|---|---|
| New pairs (493 entering) | +$623,946.94 | +318.6% |
| Lost pairs (364 exiting) | -$432,537.18 | -220.8% |
| Mix effect | +$5,893.24 | +3.0% |
| Price effect | +$1,233.60 | +0.6% |
| Volume effect | -$2,677.81 | -1.4% |
| **Total change** | **+$195,858.79** | **+38.4%** |

The net of new and lost pairs alone accounts for **+$191,409.76**, meaning the continuing matched business contributed only **+$4,449.03** in aggregate. Among **60 matched pairs** (product-customer combinations present in both years), price was barely a lever (+0.6%), volume declined (-1.4%), and mix provided modest lift (+3.0%).

---

## Deeper Insights

### 1. Volume Contraction Across Continuing Business
Total quantity **decreased** from **11,945** units in 2012 to **11,532** in 2013, a decline of **413 units (-3.46%)**. The volume effect of **-$2,677.81** confirms that existing customers bought less. The average unit price moved from **$6.4838** to **$7.1018**, a nominal improvement that generated only **$1,233.60** in price effect—modest relative to the scale of the portfolio.

### 2. Mix Shift Was Modest But Positive
The mix effect of **+$5,893.24 (+3.0%)** indicates that within continuing pairs, demand shifted toward higher-margin or higher-priced items. This partially offset the volume loss.

### 3. Top Gainers Among Matched Pairs
- **Item 09500EA | KENEXA CORP**: Quantity jumped from 6 to 62, with a 7.31% price increase, generating a **+$6,340.60** total change (predominantly mix: +$6,316.40).
- **Item 24230EA | RIGNET INC**: Quantity grew from 20 to 162 at a flat price of $40.22, contributing **+$5,711.24** (mix-driven: +$5,739.05).
- **Item 09502EA | KENEXA CORP**: Volume expanded from 3 to 35 units with a 7.19% price lift, adding **+$3,872.64**.

### 4. Top Losers Among Matched Pairs
- **Item 24230EA | CROSS COUNTRY HEALTHCAR**: Volume fell from 436 to 290 units. Despite a 1.96% price gain, total change was **-$5,867.18**, driven by a mix drag of -$5,593.68.
- **Item 09558EA | CROSS COUNTRY HEALTHCAR**: Volume declined from 80 to 63 units. A 3.16% price increase could not offset losses; total change was **-$5,108.98**.
- **Item 26173EA | AMER PUBLIC EDUCATION**: Largest single customer by base volume (2,640 units down to 1,511), also suffered a -1.96% price decline; total change **-$2,091.93**.

### 5. Customer Turnover Was Extreme
With **493 new pairs** generating **+$623,946.94** and **364 lost pairs** erasing **-$432,537.18**, the business replaced roughly 85% of its lost revenue through new acquisition. This level of churn signals either rapid market expansion or significant instability in the customer base. Notable new revenue came from Item 23541EA | SOLAZYME INC ($60,786.60), Item 37007EA | OPNEXT INC ($34,971.60), and Item 23485EA | SOLAZYME INC ($37,925.00).

---

## Data Notes

- The dataset covers **2 time periods** (2012 and 2013) with **3,000 rows** across **2,738 unique Qty Discount entities**. The panel is **unbalanced** — not all entities appear in both years.
- **Company_Level** is a constant column with only 1 unique value and carries no analytical information.
- Revenue is highly right-skewed (skewness: 12.24, mean-to-median ratio: 2.12), with the 95th percentile at $1,309.32 and a maximum of $24,579.18, indicating a long tail of high-value transactions.
- PVM decomposition covers **60 matched pairs**; the new/lost effects reflect structural mix at the portfolio boundary and should be interpreted separately from organic performance.

---

## Next Steps

1. **Investigate customer churn drivers**: 364 lost product-customer pairs costing $432,537.18 warrants a customer retention deep-dive, with attention to which customers (e.g., MERIDIAN BIOSCIENCE INC, DRIL-QUIP INC, ERESEARCHTECHNOLOGY INC) appear repeatedly in the lost list.
2. **Assess new customer quality**: 493 new pairs added $623,946.94, but sustainability is unknown. Segment by customer to determine whether new entrants are one-time or recurring buyers.
3. **Prioritize volume recovery in key accounts**: AMER PUBLIC EDUCATION and CROSS COUNTRY HEALTHCAR show meaningful volume declines in multiple SKUs; targeted account management may recover existing revenue.
4. **Evaluate pricing leverage**: At only +0.6% price effect on matched pairs, there may be room for disciplined price increases, particularly for items where volume has grown (e.g., KENEXA CORP).