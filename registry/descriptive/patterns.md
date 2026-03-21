# Descriptive Analytics — Pandas Patterns

Reference patterns for common descriptive questions. Use these as starting points, adapting column names and logic to the actual dataset.

## Distribution

```python
# Distribution of a metric by category (panel — MUST collapse to entity-level first)
# Step 1: One value per entity
member_level = df.groupby('ENTITY_COL').agg(
    avg_metric=('METRIC', 'mean'),
    category=('CATEGORY', 'first')  # use 'first' for time-invariant categories
)
# Step 2: Distribution by group on entity-level values
result = member_level.groupby('category')['avg_metric'].describe()

# Distribution of a metric (panel — no grouping)
member_level = df.groupby('ENTITY_COL')['METRIC'].mean()
result = member_level.describe()

# Distribution of a metric by category (cross-sectional)
result = df.groupby('CATEGORY')['METRIC'].describe()

# Value counts with percentages (use nunique for panel entity counts)
counts = df.groupby('CATEGORY')['ENTITY_COL'].nunique()  # panel
result = pd.DataFrame({'count': counts, 'pct': counts / counts.sum() * 100})
```

## Percentiles

```python
# Percentile by group (panel — collapse first)
member_level = df.groupby('ENTITY_COL').agg(
    avg_metric=('METRIC', 'mean'),
    group=('GROUP', 'first')
)
result = member_level.groupby('group')['avg_metric'].quantile(0.90)

# Multiple percentiles (panel — collapse first)
member_level = df.groupby('ENTITY_COL')['METRIC'].mean()
result = member_level.quantile([0.25, 0.50, 0.75, 0.90])

# Percentile by group (cross-sectional — direct)
result = df.groupby('GROUP')['METRIC'].quantile(0.90)
```

## Counts

```python
# Unique entity count (panel)
result = df['ENTITY_COL'].nunique()

# Entities meeting a condition (panel — at least one month)
result = df[df['METRIC'] == 0]['ENTITY_COL'].nunique()

# Entities meeting a condition (panel — every month)
per_entity = df.groupby('ENTITY_COL')['METRIC'].apply(lambda x: (x == 0).all())
result = per_entity.sum()
```

## Ratios & Proportions

```python
# Proportion by group (panel — collapse first)
member_level = df.groupby('ENTITY_COL').agg(
    flag=('BINARY_FLAG', 'first'),  # time-invariant flag
    group=('GROUP', 'first')
)
result = member_level.groupby('group')['flag'].mean()

# Proportion by group (cross-sectional — direct)
result = df.groupby('GROUP')['BINARY_FLAG'].mean()

# Ratio of two metrics (panel — entity-level totals)
member_level = df.groupby('ENTITY_COL').agg(
    metric_a=('METRIC_A', 'sum'),
    metric_b=('METRIC_B', 'sum'),
    group=('GROUP', 'first')
)
group_totals = member_level.groupby('group')[['metric_a', 'metric_b']].sum()
group_totals['ratio'] = group_totals['metric_a'] / group_totals['metric_b']
result = group_totals
```

## Trends Over Time (panel)

```python
# Average metric per time period
result = df.groupby('TIME_COL')['METRIC'].mean()

# Trend by category
result = df.groupby(['TIME_COL', 'CATEGORY'])['METRIC'].mean().unstack()
```

## Top-N / Ranking

```python
# Top 10 entities by metric (panel — use entity-level mean)
entity_avg = df.groupby('ENTITY_COL')['METRIC'].mean()
result = entity_avg.nlargest(10)

# Top categories
result = df.groupby('CATEGORY')['METRIC'].mean().nlargest(5)
```

## High-Cardinality Categories (>10 classes → top 10 + Other)

```python
# Panel — count by entity, not raw rows
entity_counts = df.groupby('CATEGORY')['ENTITY_COL'].nunique()
top10 = entity_counts.nlargest(10).index
df['CATEGORY_GROUPED'] = df['CATEGORY'].where(df['CATEGORY'].isin(top10), 'Other')

# Cross-sectional — count by rows
top10 = df['CATEGORY'].value_counts().nlargest(10).index
df['CATEGORY_GROUPED'] = df['CATEGORY'].where(df['CATEGORY'].isin(top10), 'Other')

# Then use CATEGORY_GROUPED in all subsequent groupby operations
member_level = df.groupby('ENTITY_COL').agg(
    avg_metric=('METRIC', 'mean'),
    category=('CATEGORY_GROUPED', 'first')
)
result = member_level.groupby('category')['avg_metric'].describe()
```
