# Metrics Definitions

## Cost Metrics

| Metric | Definition | Unit | Grain |
|--------|-----------|------|-------|
| MONTHLY_TOTAL_COST | Total allowed cost in the month | USD | member-month |
| MONTHLY_ER_COST | Emergency room allowed cost in the month | USD | member-month |
| AVG_3M_ER_COST | Rolling 3-month average of ER cost (lagged) | USD | member-month |

## Demographic Fields

| Field | Description | Type |
|-------|------------|------|
| SEX | Member sex (M/F) | Categorical |
| STATE_LATEST | Most recent state of residence | Categorical |
| COUNTY_LATEST | Most recent county of residence | Categorical |
| CITY_LATEST | Most recent city of residence | Categorical |
| MEDIAN_HOUSEHOLD_INCOME_LATEST | Census-level median household income | Numeric |

## Status Fields

| Field | Description | Values |
|-------|------------|--------|
| CURRENTLY_ACTIVE | Current enrollment status | 0=inactive, 1=active |
| ACTIVE_MONTHS | Number of months with active enrollment | Integer |

## Identifiers

| Field | Description |
|-------|------------|
| FEATURE_STORE_MEMBER_ID | Unique member identifier (entity key) |
| YEAR_MONTH_DATE | Month of observation (time dimension, YYYY-MM-DD format) |
