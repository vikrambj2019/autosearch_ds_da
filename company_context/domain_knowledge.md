# Domain Knowledge

## Healthcare Cost Analytics

This system analyzes healthcare member cost data. Key concepts:

- **Member**: An individual enrolled in a health plan, identified by FEATURE_STORE_MEMBER_ID
- **Panel data**: Monthly snapshots per member (member x month grain)
- **MONTHLY_TOTAL_COST**: Total allowed cost for a member in a given month (includes ER, inpatient, outpatient, pharmacy, etc.)
- **MONTHLY_ER_COST**: Emergency room cost component for the month
- **AVG_3M_ER_COST**: Rolling 3-month average of ER costs (lagged — useful for trend detection)
- **CURRENTLY_ACTIVE**: Whether the member is currently enrolled (1=active, 0=inactive)
- **TARGET_HIGH_COST_FLAG**: Binary flag for members in the top cost tier (used for predictive modeling)

## Cost Distribution Expectations

- Healthcare cost distributions are heavily right-skewed (many low-cost, few very high-cost)
- Median is typically much lower than mean
- Top 5% of members often account for 40-60% of total cost
- Zero-cost months are common for healthy members
- ER costs are episodic — most months are zero for most members

## Common Analytical Pitfalls

- Don't average panel rows directly as if they're independent — collapse per member first
- Active vs inactive members have very different cost profiles — always segment
- Seasonality exists in healthcare costs (flu season, open enrollment effects)
- County-level variation often reflects provider network differences, not health status
