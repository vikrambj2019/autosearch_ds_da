# Diagnostic Analytics — Pandas Patterns

## Correlation

```python
# Correlation between two metrics (panel — collapse first)
entity_level = df.groupby('ENTITY_COL').agg(
    metric_a=('METRIC_A', 'mean'),
    metric_b=('METRIC_B', 'mean')
)
corr, pval = stats.pearsonr(entity_level['metric_a'], entity_level['metric_b'])
result = f"Pearson r={corr:.3f}, p={pval:.4f}, n={len(entity_level)}"

# Correlation matrix (cross-sectional)
numeric_cols = df.select_dtypes(include='number').columns.tolist()
result = df[numeric_cols].corr()
```

## Group Comparisons

```python
# T-test: metric by binary group (panel — collapse first)
entity_level = df.groupby('ENTITY_COL').agg(
    metric=('METRIC', 'mean'),
    group=('GROUP_COL', 'first')
)
g1 = entity_level[entity_level['group'] == 'A']['metric']
g2 = entity_level[entity_level['group'] == 'B']['metric']
t_stat, p_val = stats.ttest_ind(g1, g2)
cohens_d = (g1.mean() - g2.mean()) / np.sqrt((g1.std()**2 + g2.std()**2) / 2)
result = f"t={t_stat:.3f}, p={p_val:.4f}, Cohen's d={cohens_d:.3f}, n1={len(g1)}, n2={len(g2)}"

# Mann-Whitney U (non-parametric alternative)
u_stat, p_val = stats.mannwhitneyu(g1, g2, alternative='two-sided')
result = f"U={u_stat:.0f}, p={p_val:.4f}"
```

## Anomaly / Outlier Detection

```python
# IQR method (panel — collapse to entity-level first)
member_level = df.groupby('ENTITY_COL')['METRIC'].mean()
Q1 = member_level.quantile(0.25)
Q3 = member_level.quantile(0.75)
IQR = Q3 - Q1
outlier_mask = (member_level < Q1 - 1.5*IQR) | (member_level > Q3 + 1.5*IQR)
outliers = member_level[outlier_mask]
result = f"Outlier entities: {len(outliers)} of {len(member_level)} ({len(outliers)/len(member_level)*100:.1f}%)"

# IQR method (cross-sectional — direct)
Q1 = df['METRIC'].quantile(0.25)
Q3 = df['METRIC'].quantile(0.75)
IQR = Q3 - Q1
outliers = df[(df['METRIC'] < Q1 - 1.5*IQR) | (df['METRIC'] > Q3 + 1.5*IQR)]
result = f"Outliers: {len(outliers)} of {len(df)} ({len(outliers)/len(df)*100:.1f}%)"

# Z-score method (panel — collapse first)
member_level = df.groupby('ENTITY_COL')['METRIC'].mean()
z_scores = (member_level - member_level.mean()) / member_level.std()
outliers = member_level[z_scores.abs() > 3]
```

## Threshold Exceedance

```python
# Members with METRIC_A exceeding METRIC_B by Nx
exceed = df[df['METRIC_A'] > 2 * df['METRIC_B']]
result = exceed.groupby('ENTITY_COL').size().reset_index(name='months_exceeding')
```

## Transition Analysis (panel)

```python
# Track state changes across time
df_sorted = df.sort_values(['ENTITY_COL', 'TIME_COL'])
df_sorted['prev_state'] = df_sorted.groupby('ENTITY_COL')['STATE_COL'].shift(1)
transitions = df_sorted.dropna(subset=['prev_state'])
transition_matrix = pd.crosstab(transitions['prev_state'], transitions['STATE_COL'], normalize='index')
result = transition_matrix
```

## ANOVA (multiple groups)

```python
# Panel — collapse to entity-level first
member_level = df.groupby('ENTITY_COL').agg(
    metric=('METRIC', 'mean'),
    group=('GROUP_COL', 'first')
)
groups = [g['metric'].values for _, g in member_level.groupby('group')]
f_stat, p_val = stats.f_oneway(*groups)
n_entities = len(member_level)
result = f"F={f_stat:.3f}, p={p_val:.4f}, k={len(groups)} groups, n={n_entities} entities"

# Cross-sectional — direct
groups = [g['METRIC'].values for _, g in df.groupby('GROUP_COL')]
f_stat, p_val = stats.f_oneway(*groups)
result = f"F={f_stat:.3f}, p={p_val:.4f}, k={len(groups)} groups"
```
