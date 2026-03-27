"""
Deterministic analytics computation layer.

Every function here returns structured dicts with exact numbers.
The LLM's job is ONLY to narrate these results — never to compute.

Design principles:
  - Auto-detects panel vs cross-sectional grain
  - For panel data: mean = total_sum / total_observations (no avg of avg)
  - Handles high-cardinality categories (top 10 + Other)
  - Pre-computes direction/comparison text so LLM can't get it wrong
  - All numbers are rounded and labeled — no ambiguity

Aggregation approach:
  The base unit is the OBSERVATION (one row = one member-month).
  mean = sum(metric) / count(observations) — NOT mean of per-entity means.
  This avoids the avg-of-avg trap while respecting the metric's unit (e.g., MONTHLY cost).
  Entity counts use nunique(entity_col) to avoid double-counting members.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats


# ---------------------------------------------------------------------------
# Grain helpers
# ---------------------------------------------------------------------------

def _detect_entity_time(df: pd.DataFrame) -> tuple[str | None, str | None]:
    """Lightweight grain detection: return (entity_col, time_col) or None."""
    from .data_tools import _detect_grain
    grain = _detect_grain(df)
    entity_col = grain.get("entity_col")
    time_col = grain.get("time_col")
    if entity_col and df[entity_col].nunique() == len(df):
        time_col = None  # cross-sectional
    return entity_col, time_col


def _is_panel(df: pd.DataFrame, entity_col: str | None, time_col: str | None) -> bool:
    return entity_col is not None and time_col is not None


def apply_filters(df: pd.DataFrame, filters: list[dict] | None) -> pd.DataFrame:
    """Apply a list of filter conditions to a DataFrame.

    Each filter is a dict: {"column": str, "op": str, "value": any}
    Supported ops: "==", "!=", ">", ">=", "<", "<=", "in", "not_in",
                   "contains", "not_null"

    Example:
        [{"column": "gender", "op": "==", "value": "Male"},
         {"column": "Pclass", "op": "in", "value": [1, 2]},
         {"column": "Age", "op": ">", "value": 30}]
    """
    if not filters:
        return df

    result = df.copy()
    for f in filters:
        col = f["column"]
        op = f["op"]
        val = f.get("value")

        if col not in result.columns:
            continue

        if op == "==":
            result = result[result[col] == val]
        elif op == "!=":
            result = result[result[col] != val]
        elif op == ">":
            result = result[result[col] > float(val)]
        elif op == ">=":
            result = result[result[col] >= float(val)]
        elif op == "<":
            result = result[result[col] < float(val)]
        elif op == "<=":
            result = result[result[col] <= float(val)]
        elif op == "in":
            result = result[result[col].isin(val)]
        elif op == "not_in":
            result = result[~result[col].isin(val)]
        elif op == "contains":
            result = result[result[col].astype(str).str.contains(str(val), case=False, na=False)]
        elif op == "not_null":
            result = result[result[col].notna()]

    return result


def _auto_bin_numeric_group_by(
    df: pd.DataFrame,
    gb_cols: list[str],
) -> tuple[pd.DataFrame, list[str], dict[str, str]]:
    """Auto-bin numeric group_by columns into quartiles.

    Returns (modified_df, updated_gb_cols, bin_info) where bin_info maps
    original column name to the new binned column name.

    Categorical columns pass through unchanged.
    """
    bin_info: dict[str, str] = {}
    new_gb_cols = []
    df = df.copy()
    for col in gb_cols:
        if col not in df.columns:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            bin_col = f"{col}_quartile"
            df[bin_col] = pd.qcut(df[col], q=4, duplicates="drop")
            # Format labels: "Q1 (low-high)" style
            cats = df[bin_col].cat.categories
            label_map = {}
            for i, interval in enumerate(cats):
                lo = f"{interval.left:,.0f}"
                hi = f"{interval.right:,.0f}"
                label_map[interval] = f"Q{i+1} ({lo}-{hi})"
            df[bin_col] = df[bin_col].map(label_map).astype(str)
            bin_info[col] = bin_col
            new_gb_cols.append(bin_col)
        else:
            new_gb_cols.append(col)
    return df, new_gb_cols, bin_info


def _apply_top_n(
    df: pd.DataFrame,
    category_col: str,
    entity_col: str | None = None,
    n: int = 10,
    metric_col: str | None = None,
    metric_coverage: float = 0.95,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Collapse high-cardinality categories to keep the most impactful.

    Two strategies (whichever keeps MORE categories wins):
      1. Top N by entity/row count (the original approach)
      2. Pareto: keep categories until cumulative metric coverage reaches
         threshold (default 95%), ensuring high-value segments aren't buried

    Always keeps at least 3 categories and at most `n`.
    Always returns a copy.
    """
    df = df.copy()
    n_unique = df[category_col].nunique()
    info: dict[str, Any] = {
        "original_cardinality": n_unique,
        "was_collapsed": False,
        "top_n": n,
    }

    if n_unique <= n:
        return df, info

    # Strategy 1: top N by count
    if entity_col:
        counts = df.groupby(category_col)[entity_col].nunique().sort_values(ascending=False)
    else:
        counts = df[category_col].value_counts()

    top_n_cats = set(counts.head(n).index)

    # Strategy 2: Pareto by metric (if metric provided)
    pareto_cats: set = set()
    if metric_col and metric_col in df.columns:
        metric_totals = df.groupby(category_col)[metric_col].sum().abs().sort_values(ascending=False)
        cumulative = metric_totals.cumsum()
        grand_total = metric_totals.sum()
        if grand_total > 0:
            # Keep categories until we cover metric_coverage of total
            threshold = grand_total * metric_coverage
            for cat, cum_val in cumulative.items():
                pareto_cats.add(cat)
                if cum_val >= threshold:
                    break
            # Ensure minimum of 3
            if len(pareto_cats) < 3:
                pareto_cats = set(metric_totals.head(3).index)

    # Use whichever strategy keeps more categories (capped at n)
    if pareto_cats and len(pareto_cats) > len(top_n_cats):
        # Pareto found more impactful categories
        keep_cats = set(list(pareto_cats)[:n])
        selection_method = "pareto_metric"
    else:
        keep_cats = top_n_cats
        selection_method = "top_n_count"

    top_count = counts.loc[counts.index.isin(keep_cats)].sum()
    other_count = counts.loc[~counts.index.isin(keep_cats)].sum()

    df[category_col] = df[category_col].where(
        df[category_col].isin(keep_cats), "Other"
    )

    info.update({
        "was_collapsed": True,
        "selection_method": selection_method,
        "n_kept": len(keep_cats),
        "top_categories": [c for c in counts.index if c in keep_cats],
        "top_count": int(top_count),
        "other_count": int(other_count),
        "other_n_categories": int(n_unique - len(keep_cats)),
        "note": (
            f"{category_col} had {n_unique} distinct values — "
            f"kept {len(keep_cats)} by {selection_method} + Other "
            f"({n_unique - len(keep_cats)} categories with {other_count} {'entities' if entity_col else 'rows'})."
        ),
    })

    return df, info


def _nan_report(series: pd.Series, label: str) -> dict[str, Any]:
    """Report NaN/missing data for a series."""
    total = len(series)
    n_null = int(series.isna().sum())
    return {
        "column": label,
        "total_rows": total,
        "null_count": n_null,
        "null_pct": round(n_null / total * 100, 1) if total > 0 else 0,
    }


def _confidence_interval(vals: pd.Series, confidence: float = 0.95) -> dict[str, float]:
    """Compute confidence interval for the mean."""
    n = len(vals)
    if n < 2:
        return {"ci_lower": None, "ci_upper": None, "confidence": confidence}
    mean = float(vals.mean())
    se = float(vals.std() / np.sqrt(n))
    t_crit = float(scipy_stats.t.ppf((1 + confidence) / 2, df=n - 1))
    margin = t_crit * se
    return {
        "ci_lower": round(mean - margin, 2),
        "ci_upper": round(mean + margin, 2),
        "confidence": confidence,
    }


def _safe_pct_change(new: float, old: float) -> float | None:
    """Percentage change with null instead of silent zero on division."""
    if old == 0:
        return None
    return round((new - old) / old * 100, 1)


def _group_stats(
    vals: pd.Series,
    entity_col_series: pd.Series | None = None,
) -> dict[str, Any]:
    """Compute group statistics from raw observations.

    mean = sum / count (per-observation mean, no avg of avg).
    Entity count from nunique if panel, else same as obs count.
    """
    n_obs = len(vals)
    if n_obs == 0:
        return {}

    total_sum = float(vals.sum())
    mean = total_sum / n_obs  # sum / count — the correct mean

    if entity_col_series is not None:
        n_entities = int(entity_col_series.nunique())
    else:
        n_entities = n_obs

    ci = _confidence_interval(vals)
    return {
        "entity_count": n_entities,
        "obs_count": n_obs,
        "total_sum": round(total_sum, 2),
        "mean": round(mean, 2),
        "median": round(float(vals.median()), 2),
        "std": round(float(vals.std()), 2),
        "min": round(float(vals.min()), 2),
        "p25": round(float(vals.quantile(0.25)), 2),
        "p75": round(float(vals.quantile(0.75)), 2),
        "p90": round(float(vals.quantile(0.90)), 2),
        "max": round(float(vals.max()), 2),
        "zero_pct": round(float((vals == 0).mean()) * 100, 1),
        "ci_95": ci,
    }


# ---------------------------------------------------------------------------
# 1. distribution_by_category
# ---------------------------------------------------------------------------

def distribution_by_category(
    df: pd.DataFrame,
    metric: str,
    category: str,
    entity_col: str | None = None,
    time_col: str | None = None,
    top_n: int = 10,
    filters: list[dict] | None = None,
) -> dict[str, Any]:
    """Distribution of a metric grouped by a categorical variable.

    mean = sum(metric) / count(observations) per group — no avg of avg.
    Entity counts use nunique(entity_col) for panel data.
    """
    df = apply_filters(df, filters)
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)
    data_quality = _nan_report(df[metric], metric)

    # High cardinality collapse (Pareto-aware: keeps high-cost segments)
    df, cardinality_info = _apply_top_n(df, category, entity_col, top_n, metric_col=metric)

    # Total sum (straight from raw data)
    total_sum = float(df[metric].sum())

    # Per-group stats: computed on RAW observations, not entity averages
    groups = {}
    totals = {}
    for name, grp in df.groupby(category):
        vals = grp[metric].dropna()
        entity_series = grp[entity_col] if entity_col else None
        groups[str(name)] = _group_stats(vals, entity_series)

        grp_sum = float(vals.sum())
        totals[str(name)] = {
            "total": round(grp_sum, 2),
            "cost_share_pct": round(grp_sum / total_sum * 100, 1) if total_sum > 0 else 0,
        }

    # Pre-compute comparison (highest vs lowest mean)
    comparison = None
    if len(groups) >= 2:
        sorted_groups = sorted(groups.items(), key=lambda x: x[1].get("mean", 0), reverse=True)
        highest = sorted_groups[0]
        lowest = sorted_groups[-1]
        diff_pct = _safe_pct_change(highest[1]["mean"], lowest[1]["mean"])

        comparison = {
            "higher_group": highest[0],
            "higher_mean": highest[1]["mean"],
            "lower_group": lowest[0],
            "lower_mean": lowest[1]["mean"],
            "difference_pct": diff_pct,
            "direction": (
                f"{highest[0]} has {diff_pct}% higher average {metric} "
                f"(${highest[1]['mean']:,.2f} vs ${lowest[1]['mean']:,.2f})"
            ) if diff_pct is not None else f"Cannot compare: {lowest[0]} has zero mean",
        }

    n_entities = int(df[entity_col].nunique()) if entity_col else len(df)

    return {
        "analysis_type": "distribution_by_category",
        "metric": metric,
        "category": category,
        "grain": "panel" if is_panel else "cross_sectional",
        "entity_col": entity_col,
        "entity_level_collapse": False,
        "aggregation_note": "mean = total_sum / obs_count (no avg of avg). Entity counts use nunique.",
        "n_entities": n_entities,
        "n_rows": len(df),
        "data_quality": data_quality,
        "cardinality": cardinality_info,
        "groups": groups,
        "totals": totals,
        "comparison": comparison,
        "total_sum": round(total_sum, 2),
    }


# ---------------------------------------------------------------------------
# 2. trend_over_time
# ---------------------------------------------------------------------------

def trend_over_time(
    df: pd.DataFrame,
    metric: str,
    time_col: str | None = None,
    entity_col: str | None = None,
    stratify_by: str | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Trend of a metric over time periods.

    mean = period_sum / period_obs_count — no avg of avg.
    """
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    if not time_col:
        return {"error": "No time column detected in data"}

    is_panel = _is_panel(df, entity_col, time_col)
    data_quality = _nan_report(df[metric], metric)

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    periods = sorted(df[time_col].dropna().unique())
    n_periods = len(periods)

    # Per-period: sum + count → mean = sum / count
    overall_trend = []
    for i, period in enumerate(periods):
        period_df = df[df[time_col] == period]
        period_sum = float(period_df[metric].sum())
        n_obs = len(period_df)
        period_mean = period_sum / n_obs if n_obs > 0 else 0

        if is_panel and entity_col:
            n_entities_period = int(period_df[entity_col].nunique())
        else:
            n_entities_period = n_obs

        entry = {
            "period": str(period),
            "total_sum": round(period_sum, 2),
            "n_entities": n_entities_period,
            "n_obs": n_obs,
            "mean": round(period_mean, 2),
        }
        if i > 0:
            prev_mean = overall_trend[i - 1]["mean"]
            entry["mom_change_pct"] = _safe_pct_change(entry["mean"], prev_mean)
        overall_trend.append(entry)

    # Overall direction
    if n_periods >= 2:
        first_val = overall_trend[0]["mean"]
        last_val = overall_trend[-1]["mean"]
        total_change = _safe_pct_change(last_val, first_val)
        if total_change is None:
            direction = "undefined (zero baseline)"
        elif total_change > 2:
            direction = "increasing"
        elif total_change < -2:
            direction = "decreasing"
        else:
            direction = "stable"
    else:
        total_change = None
        direction = "insufficient_data"

    # Linear regression
    regression = None
    if n_periods >= 3:
        x = np.arange(n_periods)
        y = np.array([t["mean"] for t in overall_trend])
        slope, _, r_value, p_value, _ = scipy_stats.linregress(x, y)
        regression = {
            "slope_per_period": round(float(slope), 2),
            "r_squared": round(float(r_value ** 2), 4),
            "p_value": round(float(p_value), 4),
            "note": f"Regression on {n_periods} points — interpret with caution" if n_periods < 6 else None,
        }

    result: dict[str, Any] = {
        "analysis_type": "trend_over_time",
        "metric": metric,
        "time_col": time_col,
        "grain": "panel" if is_panel else "cross_sectional",
        "aggregation_note": "mean = period_sum / obs_count (no avg of avg)",
        "n_periods": n_periods,
        "periods": [str(p) for p in periods],
        "overall_trend": overall_trend,
        "total_change_pct": total_change,
        "direction": direction,
        "regression": regression,
        "data_quality": data_quality,
        "n_entities": int(df[entity_col].nunique()) if entity_col else None,
    }

    # Stratified trends
    if stratify_by:
        df, cardinality_info = _apply_top_n(df, stratify_by, entity_col, top_n)
        result["cardinality"] = cardinality_info

        segment_trends = {}
        for seg_name, seg_df in df.groupby(stratify_by):
            seg_periods = {}
            for period in periods:
                period_slice = seg_df[seg_df[time_col] == period]
                p_sum = float(period_slice[metric].sum())
                p_n = len(period_slice)
                seg_periods[str(period)] = round(p_sum / p_n, 2) if p_n > 0 else 0

            vals = list(seg_periods.values())
            if len(vals) >= 2 and vals[0] != 0:
                pct = _safe_pct_change(vals[-1], vals[0])
            else:
                pct = None
            segment_trends[str(seg_name)] = {
                "periods": seg_periods,
                "total_change_pct": pct,
                "direction": (
                    "increasing" if pct is not None and pct > 2
                    else "decreasing" if pct is not None and pct < -2
                    else "stable" if pct is not None
                    else "undefined"
                ),
            }

        sorted_segs = sorted(
            segment_trends.items(),
            key=lambda x: abs(x[1]["total_change_pct"]) if x[1]["total_change_pct"] is not None else 0,
            reverse=True,
        )
        result["stratified_trends"] = dict(sorted_segs)
        result["fastest_growing"] = sorted_segs[0][0] if sorted_segs else None
        result["fastest_declining"] = sorted(
            segment_trends.items(),
            key=lambda x: x[1]["total_change_pct"] if x[1]["total_change_pct"] is not None else 0,
        )[0][0] if segment_trends else None

    return result


# ---------------------------------------------------------------------------
# 3. group_comparison (statistical test)
# ---------------------------------------------------------------------------

def group_comparison(
    df: pd.DataFrame,
    metric: str,
    group_col: str,
    entity_col: str | None = None,
    time_col: str | None = None,
    group_a: str | None = None,
    group_b: str | None = None,
    filters: list[dict] | None = None,
) -> dict[str, Any]:
    """Statistical comparison of a metric between groups.

    Uses raw observations — mean = sum / obs_count.
    2 groups: Welch's t-test + Mann-Whitney + Cohen's d
    3+ groups: ANOVA + Kruskal-Wallis
    """
    df = apply_filters(df, filters)
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)
    data_quality = _nan_report(df[metric], metric)

    unique_groups = sorted(df[group_col].dropna().unique(), key=str)
    n_groups = len(unique_groups)

    if n_groups < 2:
        return {"error": f"Need at least 2 groups, found {n_groups}"}

    if n_groups > 2 and group_a is None and group_b is None:
        return _multi_group_comparison(
            df, metric, group_col, unique_groups,
            is_panel, data_quality, entity_col,
        )

    if group_a is None and group_b is None:
        group_a_val, group_b_val = str(unique_groups[0]), str(unique_groups[1])
    else:
        group_a_val, group_b_val = str(group_a), str(group_b)

    return _two_group_comparison(
        df, metric, group_col,
        group_a_val, group_b_val,
        is_panel, data_quality, entity_col,
    )


def _two_group_comparison(
    df, metric, group_col,
    group_a_val, group_b_val,
    is_panel, data_quality, entity_col,
) -> dict[str, Any]:
    """Two-group comparison on raw observations."""
    vals_a = df[df[group_col].astype(str) == group_a_val][metric].dropna()
    vals_b = df[df[group_col].astype(str) == group_b_val][metric].dropna()

    if len(vals_a) < 2 or len(vals_b) < 2:
        return {"error": f"Insufficient data: group_a={len(vals_a)}, group_b={len(vals_b)}"}

    n_a, n_b = len(vals_a), len(vals_b)
    sum_a, sum_b = float(vals_a.sum()), float(vals_b.sum())
    mean_a, mean_b = sum_a / n_a, sum_b / n_b
    median_a, median_b = float(vals_a.median()), float(vals_b.median())
    std_a, std_b = float(vals_a.std()), float(vals_b.std())

    # Entity counts for panel
    if entity_col:
        grp_a_df = df[df[group_col].astype(str) == group_a_val]
        grp_b_df = df[df[group_col].astype(str) == group_b_val]
        entities_a = int(grp_a_df[entity_col].nunique())
        entities_b = int(grp_b_df[entity_col].nunique())
    else:
        entities_a, entities_b = n_a, n_b

    # Welch's t-test
    t_stat, t_pval = scipy_stats.ttest_ind(vals_a, vals_b, equal_var=False)

    # Mann-Whitney U
    u_stat, u_pval = scipy_stats.mannwhitneyu(vals_a, vals_b, alternative="two-sided")

    # Cohen's d — weighted pooled std
    pooled_std = np.sqrt(((n_a - 1) * std_a ** 2 + (n_b - 1) * std_b ** 2) / (n_a + n_b - 2))
    cohens_d = abs(mean_a - mean_b) / pooled_std if pooled_std > 0 else 0

    if cohens_d > 0.8:
        effect_label = "large"
    elif cohens_d > 0.5:
        effect_label = "medium"
    elif cohens_d > 0.2:
        effect_label = "small"
    else:
        effect_label = "negligible"

    # Direction
    if mean_a > mean_b:
        higher, lower = group_a_val, group_b_val
        higher_mean, lower_mean = mean_a, mean_b
    else:
        higher, lower = group_b_val, group_a_val
        higher_mean, lower_mean = mean_b, mean_a

    diff_pct = _safe_pct_change(higher_mean, lower_mean)
    ci_a = _confidence_interval(vals_a)
    ci_b = _confidence_interval(vals_b)

    return {
        "analysis_type": "group_comparison",
        "test_type": "two_group",
        "metric": metric,
        "group_col": group_col,
        "grain": "panel" if is_panel else "cross_sectional",
        "aggregation_note": "mean = sum / obs_count on raw observations (no avg of avg)",
        "data_quality": data_quality,
        "groups": {
            group_a_val: {
                "n_entities": entities_a, "n_obs": n_a,
                "total_sum": round(sum_a, 2),
                "mean": round(mean_a, 2), "median": round(median_a, 2),
                "std": round(std_a, 2), "ci_95": ci_a,
            },
            group_b_val: {
                "n_entities": entities_b, "n_obs": n_b,
                "total_sum": round(sum_b, 2),
                "mean": round(mean_b, 2), "median": round(median_b, 2),
                "std": round(std_b, 2), "ci_95": ci_b,
            },
        },
        "tests": {
            "welch_t_test": {
                "t_statistic": round(float(t_stat), 4),
                "p_value": round(float(t_pval), 6),
                "significant_at_05": bool(t_pval < 0.05),
                "note": "Welch's t-test (does not assume equal variances)",
            },
            "mann_whitney": {
                "u_statistic": round(float(u_stat), 2),
                "p_value": round(float(u_pval), 6),
                "significant_at_05": bool(u_pval < 0.05),
            },
            "effect_size": {
                "cohens_d": round(float(cohens_d), 4),
                "interpretation": effect_label,
                "note": "Weighted pooled std (accounts for unequal sample sizes)",
            },
        },
        "comparison": {
            "higher_group": higher,
            "lower_group": lower,
            "difference_pct": diff_pct,
            "direction": (
                f"{higher} has {diff_pct}% higher average {metric} "
                f"(${higher_mean:,.2f} vs ${lower_mean:,.2f})"
            ) if diff_pct is not None else f"Cannot compare: {lower} has zero mean",
        },
    }


def _multi_group_comparison(
    df, metric, group_col, unique_groups,
    is_panel, data_quality, entity_col,
) -> dict[str, Any]:
    """3+ group comparison on raw observations."""
    group_data = {}
    group_arrays = []
    for g in unique_groups:
        mask = df[group_col].astype(str) == str(g)
        vals = df.loc[mask, metric].dropna()
        if len(vals) < 2:
            continue
        group_arrays.append(vals)
        n_obs = len(vals)
        total = float(vals.sum())
        n_entities = int(df.loc[mask, entity_col].nunique()) if entity_col else n_obs
        ci = _confidence_interval(vals)
        group_data[str(g)] = {
            "n_entities": n_entities, "n_obs": n_obs,
            "total_sum": round(total, 2),
            "mean": round(total / n_obs, 2),
            "median": round(float(vals.median()), 2),
            "std": round(float(vals.std()), 2),
            "ci_95": ci,
        }

    if len(group_arrays) < 2:
        return {"error": f"Need at least 2 groups with sufficient data, found {len(group_arrays)}"}

    f_stat, anova_p = scipy_stats.f_oneway(*group_arrays)
    h_stat, kw_p = scipy_stats.kruskal(*group_arrays)

    # Eta-squared
    grand_mean = np.concatenate(group_arrays).mean()
    ss_between = sum(len(g) * (g.mean() - grand_mean) ** 2 for g in group_arrays)
    ss_total = sum(((g - grand_mean) ** 2).sum() for g in group_arrays)
    eta_squared = float(ss_between / ss_total) if ss_total > 0 else 0

    if eta_squared > 0.14:
        eta_label = "large"
    elif eta_squared > 0.06:
        eta_label = "medium"
    elif eta_squared > 0.01:
        eta_label = "small"
    else:
        eta_label = "negligible"

    sorted_groups = sorted(group_data.items(), key=lambda x: x[1]["mean"], reverse=True)
    highest = sorted_groups[0]
    lowest = sorted_groups[-1]
    diff_pct = _safe_pct_change(highest[1]["mean"], lowest[1]["mean"])

    return {
        "analysis_type": "group_comparison",
        "test_type": "multi_group",
        "metric": metric,
        "group_col": group_col,
        "n_groups": len(group_data),
        "grain": "panel" if is_panel else "cross_sectional",
        "aggregation_note": "mean = sum / obs_count on raw observations (no avg of avg)",
        "data_quality": data_quality,
        "groups": group_data,
        "tests": {
            "anova": {
                "f_statistic": round(float(f_stat), 4),
                "p_value": round(float(anova_p), 6),
                "significant_at_05": bool(anova_p < 0.05),
            },
            "kruskal_wallis": {
                "h_statistic": round(float(h_stat), 4),
                "p_value": round(float(kw_p), 6),
                "significant_at_05": bool(kw_p < 0.05),
            },
            "effect_size": {
                "eta_squared": round(eta_squared, 4),
                "interpretation": eta_label,
            },
        },
        "comparison": {
            "higher_group": highest[0],
            "higher_mean": highest[1]["mean"],
            "lower_group": lowest[0],
            "lower_mean": lowest[1]["mean"],
            "difference_pct": diff_pct,
            "direction": (
                f"{highest[0]} has {diff_pct}% higher average {metric} "
                f"(${highest[1]['mean']:,.2f} vs ${lowest[1]['mean']:,.2f})"
            ) if diff_pct is not None else f"Cannot compare: {lowest[0]} has zero mean",
        },
    }


# ---------------------------------------------------------------------------
# 4. correlation_analysis
# ---------------------------------------------------------------------------

def correlation_analysis(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    entity_col: str | None = None,
    time_col: str | None = None,
    filters: list[dict] | None = None,
    group_by: str | list[str] | None = None,
) -> dict[str, Any]:
    """Correlation between two numeric columns.

    Uses raw observations. For panel data, also reports entity count.
    If group_by is provided (str or list of str), runs correlation
    separately per group.
    """
    df = apply_filters(df, filters)
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)

    # Normalize group_by to list
    gb_cols = [group_by] if isinstance(group_by, str) else (group_by or [])
    gb_cols = [c for c in gb_cols if c in df.columns]

    # Auto-bin numeric group_by columns into quartiles
    if gb_cols:
        df, gb_cols, bin_info = _auto_bin_numeric_group_by(df, gb_cols)

    # Per-group correlation
    if gb_cols:
        groups = {}
        for group_key, group_df in df.groupby(gb_cols):
            key = str(group_key) if not isinstance(group_key, tuple) else (str(group_key[0]) if len(group_key) == 1 else str(group_key))
            groups[key] = _correlation_core(
                group_df, metric_a, metric_b, entity_col,
            )
        result_gb = {
            "analysis_type": "correlation",
            "metric_a": metric_a,
            "metric_b": metric_b,
            "grain": "panel" if is_panel else "cross_sectional",
            "group_by": gb_cols if len(gb_cols) > 1 else gb_cols[0],
            "groups": groups,
        }
        if bin_info:
            result_gb["binned_columns"] = bin_info
            result_gb["binning_note"] = "Numeric group_by columns were auto-binned into quartiles"
        return result_gb

    result = _correlation_core(df, metric_a, metric_b, entity_col)
    result["grain"] = "panel" if is_panel else "cross_sectional"
    result["aggregation_note"] = "computed on raw observations (one row per member-month)"
    result["data_quality"] = [
        _nan_report(df[metric_a], metric_a),
        _nan_report(df[metric_b], metric_b),
    ]
    return result


def _correlation_core(
    df: pd.DataFrame,
    metric_a: str,
    metric_b: str,
    entity_col: str | None,
) -> dict[str, Any]:
    """Core correlation computation shared by overall and per-group paths."""
    analysis_df = df[[metric_a, metric_b]].dropna()
    n_obs = len(analysis_df)

    if n_obs < 3:
        return {"error": f"Insufficient data for correlation: {n_obs} observations"}

    pearson_r, pearson_p = scipy_stats.pearsonr(analysis_df[metric_a], analysis_df[metric_b])
    spearman_r, spearman_p = scipy_stats.spearmanr(analysis_df[metric_a], analysis_df[metric_b])

    r = abs(pearson_r)
    if r > 0.7:
        strength = "strong"
    elif r > 0.4:
        strength = "moderate"
    elif r > 0.2:
        strength = "weak"
    else:
        strength = "negligible"

    sign = "positive" if pearson_r > 0 else "negative"

    n_entities = int(df[entity_col].nunique()) if entity_col else n_obs

    return {
        "analysis_type": "correlation",
        "metric_a": metric_a,
        "metric_b": metric_b,
        "n_observations": n_obs,
        "n_entities": n_entities,
        "pearson": {
            "r": round(float(pearson_r), 4),
            "p_value": round(float(pearson_p), 6),
            "significant_at_05": bool(pearson_p < 0.05),
        },
        "spearman": {
            "r": round(float(spearman_r), 4),
            "p_value": round(float(spearman_p), 6),
            "significant_at_05": bool(spearman_p < 0.05),
        },
        "interpretation": {
            "strength": strength,
            "direction": sign,
            "relationship_type": (
                "linear" if (
                    pearson_p < 0.05
                    and r >= 0.5
                    and abs(float(spearman_r)) - r < 0.15  # Spearman not much stronger → linear
                ) else "nonlinear"
            ),
            "summary": f"{strength} {sign} correlation (r={pearson_r:.4f}, p={pearson_p:.6f})",
        },
    }


# ---------------------------------------------------------------------------
# 5. summary_stats
# ---------------------------------------------------------------------------

def summary_stats(
    df: pd.DataFrame,
    metric: str,
    entity_col: str | None = None,
    time_col: str | None = None,
    filters: list[dict] | None = None,
    group_by: str | list[str] | None = None,
) -> dict[str, Any]:
    """Summary statistics for a single metric.

    Uses raw observations. mean = sum / count (no avg of avg).
    If group_by is provided (str or list of str), runs summary stats
    separately per group.
    """
    df = apply_filters(df, filters)
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)

    # Normalize group_by to list
    gb_cols = [group_by] if isinstance(group_by, str) else (group_by or [])
    gb_cols = [c for c in gb_cols if c in df.columns]

    # Auto-bin numeric group_by columns into quartiles
    if gb_cols:
        df, gb_cols, bin_info = _auto_bin_numeric_group_by(df, gb_cols)

    # Per-group summary stats
    if gb_cols:
        groups = {}
        for group_key, group_df in df.groupby(gb_cols):
            key = str(group_key) if not isinstance(group_key, tuple) else (str(group_key[0]) if len(group_key) == 1 else str(group_key))
            groups[key] = _summary_stats_core(
                group_df, metric, entity_col,
            )
        result_gb = {
            "analysis_type": "summary_stats",
            "metric": metric,
            "grain": "panel" if is_panel else "cross_sectional",
            "group_by": gb_cols if len(gb_cols) > 1 else gb_cols[0],
            "groups": groups,
        }
        if bin_info:
            result_gb["binned_columns"] = bin_info
            result_gb["binning_note"] = "Numeric group_by columns were auto-binned into quartiles"
        return result_gb

    result = _summary_stats_core(df, metric, entity_col)
    result["grain"] = "panel" if is_panel else "cross_sectional"
    result["aggregation_note"] = "mean = total_sum / obs_count on raw observations (no avg of avg)"
    n_obs = result["n_observations"]
    n_entities = result["n_entities"]
    result["computed_on"] = (
        f"raw observations ({n_obs} obs from {n_entities} entities)" if is_panel
        else f"raw rows ({n_obs})"
    )
    result["data_quality"] = _nan_report(df[metric], metric)
    return result


def _summary_stats_core(
    df: pd.DataFrame,
    metric: str,
    entity_col: str | None,
) -> dict[str, Any]:
    """Core summary stats computation shared by overall and per-group paths."""
    vals = df[metric].dropna()
    n_obs = len(vals)
    total_sum = float(vals.sum())
    mean = total_sum / n_obs if n_obs > 0 else 0
    n_entities = int(df[entity_col].nunique()) if entity_col else n_obs

    desc = vals.describe()
    ci = _confidence_interval(vals)

    std_val = float(desc["std"])
    return {
        "analysis_type": "summary_stats",
        "metric": metric,
        "n_entities": n_entities,
        "n_observations": n_obs,
        "statistics": {
            "total_sum": round(total_sum, 4),
            "mean": round(mean, 4),
            "std": round(std_val, 4),
            "min": round(float(desc["min"]), 4),
            "p25": round(float(desc["25%"]), 4),
            "median": round(float(desc["50%"]), 4),
            "p75": round(float(desc["75%"]), 4),
            "iqr": round(float(desc["75%"] - desc["25%"]), 4),
            "p90": round(float(vals.quantile(0.90)), 4),
            "p95": round(float(vals.quantile(0.95)), 4),
            "p99": round(float(vals.quantile(0.99)), 4),
            "max": round(float(desc["max"]), 4),
            "ci_95": ci,
        },
        "shape": {
            "skewness": round(float(vals.skew()), 4),
            "kurtosis": round(float(vals.kurtosis()), 4),
            "kurtosis_regular": round(float(vals.kurtosis()) + 3, 4),
            "scipy_skewness": round(float(scipy_stats.skew(vals, bias=True)), 4),
            "scipy_kurtosis_excess": round(float(scipy_stats.kurtosis(vals, bias=True)), 4),
            "scipy_kurtosis_regular": round(float(scipy_stats.kurtosis(vals, bias=True)) + 3, 4),
            "values_within_1_std": int(((vals >= mean - std_val) & (vals <= mean + std_val)).sum()),
            "values_within_2_std": int(((vals >= mean - 2 * std_val) & (vals <= mean + 2 * std_val)).sum()),
            "zero_pct": round(float((vals == 0).mean()) * 100, 1),
            "mean_to_median_ratio": round(mean / float(desc["50%"]), 4) if desc["50%"] > 0 else None,
        },
    }


# ---------------------------------------------------------------------------
# 5b. normality_test
# ---------------------------------------------------------------------------

def normality_test(
    df: pd.DataFrame,
    metric: str,
    group_col: str | None = None,
    entity_col: str | None = None,
    time_col: str | None = None,
    alpha: float = 0.05,
    filters: list[dict] | None = None,
) -> dict[str, Any]:
    """Test whether a column follows a normal distribution.

    Runs Shapiro-Wilk and Kolmogorov-Smirnov tests. Returns pre-computed
    decision so the LLM cannot get the interpretation wrong.

    If group_col is provided, runs the test separately for each group.
    """
    df = apply_filters(df, filters)
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)

    def _test_series(vals: pd.Series) -> dict:
        vals = vals.dropna()
        n = len(vals)
        if n < 8:
            return {"error": f"Insufficient data for normality test: {n} values"}

        # Shapiro-Wilk (best for n < 5000)
        if n <= 5000:
            sw_stat, sw_p = scipy_stats.shapiro(vals)
        else:
            # Sample for Shapiro (it has a 5000 limit in some implementations)
            sample = vals.sample(n=5000, random_state=42)
            sw_stat, sw_p = scipy_stats.shapiro(sample)

        # Kolmogorov-Smirnov (against normal with same mean/std)
        ks_stat, ks_p = scipy_stats.kstest(vals, "norm", args=(vals.mean(), vals.std()))

        # Anderson-Darling
        ad_result = scipy_stats.anderson(vals, dist="norm")
        # Use the 5% significance level (index 2)
        ad_stat = float(ad_result.statistic)
        ad_critical_5pct = float(ad_result.critical_values[2])
        ad_is_normal = ad_stat < ad_critical_5pct

        # Pre-computed decision: use Shapiro-Wilk as primary
        is_normal = bool(sw_p >= alpha)

        # Skewness and kurtosis for context
        skewness = round(float(scipy_stats.skew(vals, bias=True)), 4)
        kurtosis_excess = round(float(scipy_stats.kurtosis(vals, bias=True)), 4)
        kurtosis = kurtosis_excess  # excess kurtosis (default in scipy)
        kurtosis_regular = round(kurtosis_excess + 3, 4)  # Fisher → Pearson

        return {
            "n": n,
            "is_normal": is_normal,
            "decision": "normally distributed" if is_normal else "not normally distributed",
            "alpha": alpha,
            "shapiro_wilk": {
                "statistic": round(float(sw_stat), 4),
                "p_value": round(float(sw_p), 6),
                "is_normal": bool(sw_p >= alpha),
            },
            "kolmogorov_smirnov": {
                "statistic": round(float(ks_stat), 4),
                "p_value": round(float(ks_p), 6),
                "is_normal": bool(ks_p >= alpha),
            },
            "anderson_darling": {
                "statistic": round(ad_stat, 4),
                "critical_value_5pct": round(ad_critical_5pct, 4),
                "is_normal": ad_is_normal,
            },
            "skewness": skewness,
            "kurtosis_excess": kurtosis_excess,
            "kurtosis_regular": kurtosis_regular,
        }

    result: dict[str, Any] = {
        "analysis_type": "normality_test",
        "metric": metric,
        "grain": "panel" if is_panel else "cross_sectional",
    }

    if group_col:
        groups = {}
        for group_name, group_df in df.groupby(group_col):
            if metric in group_df.columns:
                groups[str(group_name)] = _test_series(group_df[metric])
        result["groups"] = groups
        result["group_col"] = group_col
    else:
        result.update(_test_series(df[metric]))

    return result


# ---------------------------------------------------------------------------
# 5c. entity_lookup
# ---------------------------------------------------------------------------

def entity_lookup(
    df: pd.DataFrame,
    metric: str,
    category: str,
    mode: str = "max",
    top_n: int = 5,
    entity_col: str | None = None,
    time_col: str | None = None,
    filters: list[dict] | None = None,
) -> dict[str, Any]:
    """Find which entity has the highest/lowest value of a metric.

    Answers questions like "which country has the highest happiness score?"
    or "which store has the lowest revenue?"

    For panel data, aggregates to entity-level (mean) before ranking.

    Args:
        metric: Numeric column to rank by
        category: Column containing the entity names to return
        mode: "max" or "min"
        top_n: Number of top/bottom entries to return
    """
    df = apply_filters(df, filters)
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)

    # For panel data, aggregate to one row per category value
    if is_panel and category != entity_col:
        agg_df = df.groupby(category)[metric].mean().reset_index()
    else:
        agg_df = df[[category, metric]].dropna()

    if len(agg_df) == 0:
        return {"error": f"No data for {metric} grouped by {category}"}

    # Sort
    ascending = mode == "min"
    ranked = agg_df.sort_values(metric, ascending=ascending).reset_index(drop=True)

    # Top entry
    top_row = ranked.iloc[0]
    answer = str(top_row[category])
    answer_value = round(float(top_row[metric]), 4)

    # Leaderboard
    n = min(top_n, len(ranked))
    leaderboard = [
        {"rank": i + 1, category: str(ranked.iloc[i][category]),
         metric: round(float(ranked.iloc[i][metric]), 4)}
        for i in range(n)
    ]

    return {
        "analysis_type": "entity_lookup",
        "question_type": f"which {category} has the {'highest' if mode == 'max' else 'lowest'} {metric}",
        "metric": metric,
        "category": category,
        "mode": mode,
        "grain": "panel" if is_panel else "cross_sectional",
        "answer": answer,
        "answer_value": answer_value,
        "n_entities": int(agg_df[category].nunique()),
        "leaderboard": leaderboard,
    }


# ---------------------------------------------------------------------------
# 6. entity_counts
# ---------------------------------------------------------------------------

def entity_counts(
    df: pd.DataFrame,
    group_col: str | None = None,
    entity_col: str | None = None,
    time_col: str | None = None,
    top_n: int = 10,
) -> dict[str, Any]:
    """Count unique entities, optionally grouped by a category."""
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)

    if entity_col:
        total = int(df[entity_col].nunique())
    else:
        total = len(df)

    result: dict[str, Any] = {
        "analysis_type": "entity_counts",
        "entity_col": entity_col,
        "grain": "panel" if is_panel else "cross_sectional",
        "total_entities": total,
        "total_rows": len(df),
    }

    if group_col:
        df_work, cardinality_info = _apply_top_n(df, group_col, entity_col, top_n)
        result["cardinality"] = cardinality_info

        if entity_col:
            counts = df_work.groupby(group_col)[entity_col].nunique().sort_values(ascending=False)
        else:
            counts = df_work[group_col].value_counts()

        result["groups"] = {
            str(k): {"count": int(v), "pct": round(float(v / total * 100), 1)}
            for k, v in counts.items()
        }

        result["groups_sum"] = int(counts.sum())
        if result["groups_sum"] != total:
            result["note"] = (
                f"Group counts sum to {result['groups_sum']} but total entities = {total}. "
                f"Some entities appear in multiple groups (e.g., status changes over time)."
            )

    return result


# ---------------------------------------------------------------------------
# 7. price_volume_mix — PVM decomposition between two periods
# ---------------------------------------------------------------------------

def price_volume_mix(
    df: pd.DataFrame,
    revenue_col: str = "rev",
    qty_col: str = "qty",
    product_col: str = "product_sku",
    customer_col: str | None = None,
    time_col: str | None = None,
    period_a: str | None = None,
    period_b: str | None = None,
    cost_col: str | None = None,
    margin_col: str | None = None,
    dimensions: list[str] | None = None,
    filters: list[dict] | None = None,
) -> dict[str, Any]:
    """Price-Volume-Mix decomposition of revenue (or margin) change between two periods.

    Decomposes total change into:
      - Price effect: unit price changes on matched product-customer pairs
      - Volume effect: total quantity change at base-period prices and mix
      - Mix effect: shift in product/customer mix at base-period prices
      - New effect: revenue from new product-customer combos (only in period B)
      - Lost effect: revenue lost from churned combos (only in period A)

    Auto-detects pricing grain: if unit prices vary by customer within the same
    product_sku, matches at (product_sku, customer_id) level. Otherwise matches
    at product_sku level only.
    """
    df = apply_filters(df, filters)
    df = df.copy()

    # ── Detect time column and resolve periods ──────────────────────────
    if time_col is None:
        # Auto-detect: date-like column names first, then integer year columns
        time_keywords = ("date", "period", "month", "year", "time")
        for c in df.columns:
            if any(kw in c.lower() for kw in time_keywords):
                time_col = c
                break
        if time_col is None:
            return {"error": "No time/date column detected. Pass time_col explicitly."}

    # Keep integer year columns as-is (don't convert 2025/2026 via pd.to_datetime
    # which would interpret them as nanoseconds from epoch)
    col_is_int_year = (
        pd.api.types.is_integer_dtype(df[time_col])
        and df[time_col].dropna().between(1900, 2100).all()
    )
    if not col_is_int_year and not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    periods = sorted(df[time_col].dropna().unique())
    if len(periods) < 2:
        return {"error": f"Need at least 2 periods, found {len(periods)}"}

    if period_a is None and period_b is None:
        pa, pb = periods[-2], periods[-1]
    else:
        pa = _match_period(period_a, periods) if period_a else periods[-2]
        pb = _match_period(period_b, periods) if period_b else periods[-1]

    df_a = df[df[time_col] == pa].copy()
    df_b = df[df[time_col] == pb].copy()

    if len(df_a) == 0 or len(df_b) == 0:
        return {"error": f"No data for one of the periods: {pa}, {pb}"}

    # ── Auto-detect pricing grain ───────────────────────────────────────
    # Check if unit price varies by customer within the same product
    customer_pricing = False
    if customer_col and customer_col in df.columns:
        df_a["_unit_price"] = df_a[revenue_col] / df_a[qty_col].replace(0, np.nan)
        price_cv = df_a.groupby(product_col)["_unit_price"].apply(
            lambda x: x.std() / x.mean() if x.mean() > 0 and len(x) > 1 else 0
        )
        # If >10% of products have CV > 1%, pricing is customer-specific
        if (price_cv > 0.01).mean() > 0.1:
            customer_pricing = True

    # ── Build match key ─────────────────────────────────────────────────
    if customer_pricing and customer_col:
        key_cols = [product_col, customer_col]
        grain_label = f"{product_col} x {customer_col}"
    else:
        key_cols = [product_col]
        grain_label = product_col

    def _make_key(row):
        return tuple(str(row[c]) for c in key_cols) if len(key_cols) > 1 else str(row[key_cols[0]])

    # ── Aggregate to grain level per period ─────────────────────────────
    agg_cols = {qty_col: "sum", revenue_col: "sum"}
    if cost_col and cost_col in df.columns:
        agg_cols[cost_col] = "sum"
    if margin_col and margin_col in df.columns:
        agg_cols[margin_col] = "sum"

    grp_a = df_a.groupby(key_cols, as_index=False).agg(agg_cols)
    grp_b = df_b.groupby(key_cols, as_index=False).agg(agg_cols)

    # Compute unit prices
    grp_a["_unit_price"] = grp_a[revenue_col] / grp_a[qty_col].replace(0, np.nan)
    grp_b["_unit_price"] = grp_b[revenue_col] / grp_b[qty_col].replace(0, np.nan)

    if margin_col and margin_col in grp_a.columns:
        grp_a["_unit_margin"] = grp_a[margin_col] / grp_a[qty_col].replace(0, np.nan)
        grp_b["_unit_margin"] = grp_b[margin_col] / grp_b[qty_col].replace(0, np.nan)

    # ── Match / New / Lost ──────────────────────────────────────────────
    grp_a["_key"] = grp_a.apply(_make_key, axis=1)
    grp_b["_key"] = grp_b.apply(_make_key, axis=1)

    keys_a = set(grp_a["_key"])
    keys_b = set(grp_b["_key"])
    matched_keys = keys_a & keys_b
    new_keys = keys_b - keys_a
    lost_keys = keys_a - keys_b

    matched_a = grp_a[grp_a["_key"].isin(matched_keys)].set_index("_key")
    matched_b = grp_b[grp_b["_key"].isin(matched_keys)].set_index("_key")
    new_rows = grp_b[grp_b["_key"].isin(new_keys)]
    lost_rows = grp_a[grp_a["_key"].isin(lost_keys)]

    # ── PVM decomposition (Laspeyres — base-period weights) ─────────────
    # Align matched data
    matched_b = matched_b.reindex(matched_a.index)

    q_a = matched_a[qty_col].values
    q_b = matched_b[qty_col].values
    p_a = matched_a["_unit_price"].values
    p_b = matched_b["_unit_price"].values

    Q_a = q_a.sum()  # total base quantity
    Q_b = q_b.sum()  # total current quantity

    # Base-period mix weights
    w_a = q_a / Q_a if Q_a > 0 else np.zeros_like(q_a)

    # Weighted average base price (using base mix)
    P_bar_a = float(np.sum(w_a * p_a))

    # 1. Volume effect: change in total quantity × base weighted-avg price
    volume_effect = float((Q_b - Q_a) * P_bar_a)

    # 2. Price effect: Σ (price_change_i × base_qty_i)
    price_effect = float(np.nansum((p_b - p_a) * q_a))

    # 3. Mix effect = matched_revenue_change - volume - price
    matched_rev_a = float(matched_a[revenue_col].sum())
    matched_rev_b = float(matched_b[revenue_col].sum())
    matched_change = matched_rev_b - matched_rev_a
    mix_effect = matched_change - volume_effect - price_effect

    # 4. New / Lost effects
    new_effect = float(new_rows[revenue_col].sum()) if len(new_rows) > 0 else 0.0
    lost_effect = float(-lost_rows[revenue_col].sum()) if len(lost_rows) > 0 else 0.0

    total_rev_a = float(grp_a[revenue_col].sum())
    total_rev_b = float(grp_b[revenue_col].sum())
    total_change = total_rev_b - total_rev_a

    # ── Per-product breakdown ───────────────────────────────────────────
    # Build lookup dicts for safe access (avoids MultiIndex .loc issues with tuples)
    _lookup_a = {k: i for i, k in enumerate(matched_a.index)}
    _lookup_b = {k: i for i, k in enumerate(matched_b.index)}

    product_details = []
    for key in sorted(matched_keys):
        idx_a = _lookup_a[key]
        idx_b = _lookup_b[key]
        qa_i = float(matched_a.iloc[idx_a][qty_col])
        qb_i = float(matched_b.iloc[idx_b][qty_col])
        pa_i = float(matched_a.iloc[idx_a]["_unit_price"])
        pb_i = float(matched_b.iloc[idx_b]["_unit_price"])
        rev_a_i = float(matched_a.iloc[idx_a][revenue_col])
        rev_b_i = float(matched_b.iloc[idx_b][revenue_col])

        price_i = (pb_i - pa_i) * qa_i
        vol_share_a = qa_i / Q_a if Q_a > 0 else 0
        vol_i = (Q_b - Q_a) * vol_share_a * pa_i
        mix_i = (rev_b_i - rev_a_i) - price_i - vol_i

        detail = {
            "key": key if isinstance(key, str) else " | ".join(key) if isinstance(key, tuple) else str(key),
            "qty_base": round(qa_i, 2),
            "qty_current": round(qb_i, 2),
            "unit_price_base": round(pa_i, 4),
            "unit_price_current": round(pb_i, 4),
            "price_change_pct": round((pb_i / pa_i - 1) * 100, 2) if pa_i > 0 else None,
            "rev_base": round(rev_a_i, 2),
            "rev_current": round(rev_b_i, 2),
            "price_effect": round(price_i, 2),
            "volume_effect": round(vol_i, 2),
            "mix_effect": round(mix_i, 2),
            "total_change": round(rev_b_i - rev_a_i, 2),
        }
        product_details.append(detail)

    # Sort by absolute total change descending
    product_details.sort(key=lambda x: abs(x["total_change"]), reverse=True)

    # ── Margin PVM (if margin column available) ─────────────────────────
    margin_pvm = None
    if margin_col and margin_col in matched_a.columns:
        m_a = matched_a["_unit_margin"].values
        m_b = matched_b["_unit_margin"].values

        M_bar_a = float(np.sum(w_a * m_a))
        margin_volume = float((Q_b - Q_a) * M_bar_a)
        margin_price = float(np.nansum((m_b - m_a) * q_a))
        matched_margin_a = float(matched_a[margin_col].sum())
        matched_margin_b = float(matched_b[margin_col].sum())
        margin_mix = (matched_margin_b - matched_margin_a) - margin_volume - margin_price

        new_margin = float(new_rows[margin_col].sum()) if (len(new_rows) > 0 and margin_col in new_rows.columns) else 0.0
        lost_margin = float(-lost_rows[margin_col].sum()) if (len(lost_rows) > 0 and margin_col in lost_rows.columns) else 0.0
        total_margin_a = float(grp_a[margin_col].sum())
        total_margin_b = float(grp_b[margin_col].sum())

        margin_pvm = {
            "metric": margin_col,
            "total_base": round(total_margin_a, 2),
            "total_current": round(total_margin_b, 2),
            "total_change": round(total_margin_b - total_margin_a, 2),
            "volume_effect": round(margin_volume, 2),
            "price_effect": round(margin_price, 2),
            "mix_effect": round(margin_mix, 2),
            "new_effect": round(new_margin, 2),
            "lost_effect": round(lost_margin, 2),
        }

    # ── Dimensional drivers ─────────────────────────────────────────────
    # Auto-detect categorical dimensions for driver analysis
    if dimensions is None:
        dimensions = []
        for c in df.columns:
            if c in (time_col, qty_col, revenue_col, cost_col, margin_col):
                continue
            if c in key_cols:
                continue
            if df[c].dtype == "object" and 1 < df[c].nunique() <= 50:
                dimensions.append(c)

    dim_drivers = {}
    for dim in dimensions:
        if dim not in df.columns:
            continue
        # Aggregate by dimension for each period
        dim_a = df_a.groupby(dim).agg({revenue_col: "sum", qty_col: "sum"}).rename(
            columns={revenue_col: "rev_base", qty_col: "qty_base"}
        )
        dim_b = df_b.groupby(dim).agg({revenue_col: "sum", qty_col: "sum"}).rename(
            columns={revenue_col: "rev_current", qty_col: "qty_current"}
        )
        dim_merged = dim_a.join(dim_b, how="outer").fillna(0)
        dim_merged["change"] = dim_merged["rev_current"] - dim_merged["rev_base"]
        dim_merged["pct_of_total_change"] = (
            (dim_merged["change"] / total_change * 100) if total_change != 0 else 0
        )

        segments = []
        for idx_val, row in dim_merged.iterrows():
            segments.append({
                dim: str(idx_val),
                "rev_base": round(float(row["rev_base"]), 2),
                "rev_current": round(float(row["rev_current"]), 2),
                "change": round(float(row["change"]), 2),
                "pct_of_total_change": round(float(row["pct_of_total_change"]), 1),
            })
        segments.sort(key=lambda x: abs(x["change"]), reverse=True)
        dim_drivers[dim] = segments

    # ── New & lost product details ──────────────────────────────────────
    def _format_key(k):
        if isinstance(k, tuple):
            return " | ".join(str(x) for x in k)
        return str(k)

    new_details = []
    for _, row in new_rows.iterrows():
        new_details.append({
            "key": _format_key(row["_key"]),
            "qty": round(float(row[qty_col]), 2),
            "revenue": round(float(row[revenue_col]), 2),
        })

    lost_details = []
    for _, row in lost_rows.iterrows():
        lost_details.append({
            "key": _format_key(row["_key"]),
            "qty": round(float(row[qty_col]), 2),
            "revenue": round(float(row[revenue_col]), 2),
        })

    # ── Assemble result ─────────────────────────────────────────────────
    def _pct(val, total):
        return round(val / total * 100, 1) if total != 0 else 0.0

    result: dict[str, Any] = {
        "analysis_type": "price_volume_mix",
        "period_a": str(pa),
        "period_b": str(pb),
        "pricing_grain": grain_label,
        "customer_level_pricing": customer_pricing,
        "method": "Laspeyres (base-period weights)",
        "matched_pairs": len(matched_keys),
        "new_pairs": len(new_keys),
        "lost_pairs": len(lost_keys),
        "aggregate": {
            "revenue_base": round(total_rev_a, 2),
            "revenue_current": round(total_rev_b, 2),
            "total_change": round(total_change, 2),
            "total_change_pct": round((total_rev_b / total_rev_a - 1) * 100, 2) if total_rev_a > 0 else None,
            "volume_effect": round(volume_effect, 2),
            "volume_pct": _pct(volume_effect, total_change),
            "price_effect": round(price_effect, 2),
            "price_pct": _pct(price_effect, total_change),
            "mix_effect": round(mix_effect, 2),
            "mix_pct": _pct(mix_effect, total_change),
            "new_effect": round(new_effect, 2),
            "new_pct": _pct(new_effect, total_change),
            "lost_effect": round(lost_effect, 2),
            "lost_pct": _pct(lost_effect, total_change),
            "direction": "increased" if total_change > 0 else "decreased" if total_change < 0 else "unchanged",
        },
        "qty_summary": {
            "total_base": round(float(Q_a), 2),
            "total_current": round(float(Q_b), 2),
            "change": round(float(Q_b - Q_a), 2),
            "change_pct": round((Q_b / Q_a - 1) * 100, 2) if Q_a > 0 else None,
        },
        "avg_unit_price": {
            "base": round(P_bar_a, 4),
            "current": round(float(np.sum(q_b / Q_b * p_b)) if Q_b > 0 else 0, 4),
        },
        "product_detail": product_details,
        "new_products": new_details,
        "lost_products": lost_details,
        "dimensional_drivers": dim_drivers,
    }

    if margin_pvm:
        result["margin_pvm"] = margin_pvm

    return result


# ---------------------------------------------------------------------------
# 8. period_comparison (entity-level diff between two time periods)
# ---------------------------------------------------------------------------

def period_comparison(
    df: pd.DataFrame,
    metric: str,
    period_a: str | None = None,
    period_b: str | None = None,
    entity_col: str | None = None,
    time_col: str | None = None,
    stratify_by: str | None = None,
    top_n: int = 10,
    top_movers: int = 10,
) -> dict[str, Any]:
    """Entity-level comparison between two time periods.

    This is the analysis you can ONLY do with panel data:
      - Which members got more expensive / cheaper?
      - Who are the movers (jumped cost buckets)?
      - Net new vs churned members between periods
      - Which segments drove the aggregate change?

    mean = sum / count (no avg of avg). Entity matching via inner join.
    """
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    if not time_col:
        return {"error": "No time column detected — period_comparison requires panel data"}
    if not entity_col:
        return {"error": "No entity column detected — period_comparison requires panel data"}

    df = df.copy()
    if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
        df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    periods = sorted(df[time_col].dropna().unique())
    if len(periods) < 2:
        return {"error": f"Need at least 2 periods, found {len(periods)}"}

    # Resolve period_a and period_b
    period_labels = [str(p) for p in periods]
    if period_a is None and period_b is None:
        # Default: last two periods
        pa, pb = periods[-2], periods[-1]
    elif period_a is not None and period_b is None:
        # Find the period matching period_a, use next one
        pa = _match_period(period_a, periods)
        idx = list(periods).index(pa)
        if idx + 1 < len(periods):
            pb = periods[idx + 1]
        else:
            return {"error": f"No period after {pa}"}
    else:
        pa = _match_period(period_a, periods)
        pb = _match_period(period_b, periods)

    df_a = df[df[time_col] == pa][[entity_col, metric]].dropna()
    df_b = df[df[time_col] == pb][[entity_col, metric]].dropna()

    entities_a = set(df_a[entity_col])
    entities_b = set(df_b[entity_col])

    matched = entities_a & entities_b
    new_entities = entities_b - entities_a
    churned_entities = entities_a - entities_b

    # ── 1. Aggregate delta ─────────────────────────────────────────────
    sum_a, sum_b = float(df_a[metric].sum()), float(df_b[metric].sum())
    n_obs_a, n_obs_b = len(df_a), len(df_b)
    mean_a = sum_a / n_obs_a if n_obs_a > 0 else 0
    mean_b = sum_b / n_obs_b if n_obs_b > 0 else 0
    delta_pct = _safe_pct_change(mean_b, mean_a)

    if delta_pct is not None and delta_pct > 2:
        agg_direction = "increasing"
    elif delta_pct is not None and delta_pct < -2:
        agg_direction = "decreasing"
    else:
        agg_direction = "stable"

    aggregate = {
        "period_a": str(pa),
        "period_b": str(pb),
        "period_a_stats": {
            "n_entities": len(entities_a),
            "n_obs": n_obs_a,
            "total_sum": round(sum_a, 2),
            "mean": round(mean_a, 2),
        },
        "period_b_stats": {
            "n_entities": len(entities_b),
            "n_obs": n_obs_b,
            "total_sum": round(sum_b, 2),
            "mean": round(mean_b, 2),
        },
        "delta_mean": round(mean_b - mean_a, 2),
        "delta_pct": delta_pct,
        "delta_sum": round(sum_b - sum_a, 2),
        "direction": agg_direction,
        "direction_text": (
            f"Average {metric} {'increased' if mean_b > mean_a else 'decreased'} "
            f"by {abs(delta_pct)}% from ${mean_a:,.2f} to ${mean_b:,.2f}"
        ) if delta_pct is not None else "Cannot compute change (zero baseline)",
    }

    # ── 2. Cohort matching ─────────────────────────────────────────────
    cohort = {
        "matched_entities": len(matched),
        "new_entities": len(new_entities),
        "churned_entities": len(churned_entities),
        "retention_rate": round(len(matched) / len(entities_a) * 100, 1) if entities_a else 0,
        "note": (
            f"{len(matched)} members in both periods, "
            f"{len(new_entities)} new in {str(pb)[:10]}, "
            f"{len(churned_entities)} absent from {str(pb)[:10]}"
        ),
    }

    # Cost of new and churned
    if new_entities:
        new_vals = df_b[df_b[entity_col].isin(new_entities)][metric]
        cohort["new_entity_mean"] = round(float(new_vals.mean()), 2)
        cohort["new_entity_sum"] = round(float(new_vals.sum()), 2)
    if churned_entities:
        churned_vals = df_a[df_a[entity_col].isin(churned_entities)][metric]
        cohort["churned_entity_mean"] = round(float(churned_vals.mean()), 2)
        cohort["churned_entity_sum"] = round(float(churned_vals.sum()), 2)

    # ── 3. Member movement (matched cohort only) ───────────────────────
    movement: dict[str, Any] = {}
    per_entity_change = pd.Series(dtype=float)
    if matched:
        matched_a = df_a[df_a[entity_col].isin(matched)].set_index(entity_col)[metric]
        matched_b = df_b[df_b[entity_col].isin(matched)].set_index(entity_col)[metric]
        # Align and compute per-entity change
        per_entity_change = (matched_b - matched_a).dropna()

        n_matched = len(per_entity_change)
        n_increased = int((per_entity_change > 0).sum())
        n_decreased = int((per_entity_change < 0).sum())
        n_flat = int((per_entity_change == 0).sum())

        movement = {
            "n_matched": n_matched,
            "n_increased": n_increased,
            "pct_increased": round(n_increased / n_matched * 100, 1) if n_matched > 0 else 0,
            "n_decreased": n_decreased,
            "pct_decreased": round(n_decreased / n_matched * 100, 1) if n_matched > 0 else 0,
            "n_flat": n_flat,
            "pct_flat": round(n_flat / n_matched * 100, 1) if n_matched > 0 else 0,
            "mean_change_per_entity": round(float(per_entity_change.mean()), 2),
            "median_change_per_entity": round(float(per_entity_change.median()), 2),
            "std_change": round(float(per_entity_change.std()), 2),
            "direction": (
                f"Of {n_matched} matched members, "
                f"{n_increased} ({round(n_increased/n_matched*100,1)}%) increased, "
                f"{n_decreased} ({round(n_decreased/n_matched*100,1)}%) decreased, "
                f"{n_flat} ({round(n_flat/n_matched*100,1)}%) unchanged"
            ) if n_matched > 0 else "No matched members",
        }

    # ── 4. Transition matrix (cost buckets) ────────────────────────────
    transition = {}
    if matched and len(per_entity_change) > 0:
        # Build buckets from overall distribution
        all_vals = pd.concat([df_a[metric], df_b[metric]]).dropna()
        p50 = float(all_vals.median())
        p90 = float(all_vals.quantile(0.90))

        def _bucket(v: float) -> str:
            if v == 0:
                return "$0"
            elif v <= p50:
                return f"$1-${p50:,.0f}"
            elif v <= p90:
                return f"${p50:,.0f}-${p90:,.0f}"
            else:
                return f"${p90:,.0f}+"

        bucket_labels = ["$0", f"$1-${p50:,.0f}", f"${p50:,.0f}-${p90:,.0f}", f"${p90:,.0f}+"]

        matched_a = df_a[df_a[entity_col].isin(matched)].set_index(entity_col)[metric]
        matched_b = df_b[df_b[entity_col].isin(matched)].set_index(entity_col)[metric]

        buckets_a = matched_a.map(_bucket)
        buckets_b = matched_b.map(_bucket)

        # Cross-tabulation
        cross = pd.crosstab(buckets_a, buckets_b, dropna=False)
        # Reindex to ensure all buckets appear
        cross = cross.reindex(index=bucket_labels, columns=bucket_labels, fill_value=0)

        transition = {
            "bucket_thresholds": {"p50": round(p50, 2), "p90": round(p90, 2)},
            "bucket_labels": bucket_labels,
            "matrix": {
                str(row): {str(col): int(cross.loc[row, col]) for col in cross.columns}
                for row in cross.index
            },
            "stayed_same_bucket": int(sum(cross.loc[b, b] for b in bucket_labels if b in cross.index and b in cross.columns)),
            "moved_up": 0,
            "moved_down": 0,
        }

        # Count moves up/down
        for i, row_b in enumerate(bucket_labels):
            for j, col_b in enumerate(bucket_labels):
                if row_b in cross.index and col_b in cross.columns:
                    count = int(cross.loc[row_b, col_b])
                    if j > i:
                        transition["moved_up"] += count
                    elif j < i:
                        transition["moved_down"] += count

        transition["direction"] = (
            f"{transition['stayed_same_bucket']} stayed in same bucket, "
            f"{transition['moved_up']} moved to higher cost, "
            f"{transition['moved_down']} moved to lower cost"
        )

    # ── 5. Top movers ─────────────────────────────────────────────────
    top_movers_result: dict[str, Any] = {}
    if len(per_entity_change) > 0:
        sorted_changes = per_entity_change.sort_values()
        top_increases = sorted_changes.tail(top_movers).sort_values(ascending=False)
        top_decreases = sorted_changes.head(top_movers)

        top_movers_result = {
            "biggest_increases": [
                {"entity": str(eid), "change": round(float(val), 2)}
                for eid, val in top_increases.items() if val > 0
            ],
            "biggest_decreases": [
                {"entity": str(eid), "change": round(float(val), 2)}
                for eid, val in top_decreases.items() if val < 0
            ],
        }

    # ── 6. Auto driver analysis (all categorical columns) ────────────
    auto_drivers: dict[str, Any] = {}
    if matched and len(per_entity_change) > 0:
        # Compute per-entity change for matched cohort
        matched_a_s = df_a[df_a[entity_col].isin(matched)].set_index(entity_col)[metric]
        matched_b_s = df_b[df_b[entity_col].isin(matched)].set_index(entity_col)[metric]
        change_series = (matched_b_s - matched_a_s).dropna()
        total_delta = float(change_series.sum())

        # Build entity-level attribute lookup from full df (latest value per entity)
        df_sorted = df.sort_values(time_col)
        entity_attrs = df_sorted.drop_duplicates(entity_col, keep="last").set_index(entity_col)

        # Find all categorical columns (exclude entity, time, numerics)
        cat_cols = [
            c for c in df.columns
            if c != entity_col and c != time_col
            and not pd.api.types.is_numeric_dtype(df[c])
            and df[c].nunique() > 1  # skip constants
        ]

        # Optionally add user-specified stratify_by even if numeric
        if stratify_by and stratify_by in df.columns and stratify_by not in cat_cols:
            cat_cols.append(stratify_by)

        by_column: dict[str, Any] = {}
        all_drivers: list[dict[str, Any]] = []
        min_contribution_pct = 5.0  # filter threshold

        for col in cat_cols:
            if col not in entity_attrs.columns:
                continue

            # Map each matched entity to its category value
            col_map = entity_attrs[col]

            # Apply top N collapse for high-cardinality
            unique_vals = col_map.loc[col_map.index.isin(change_series.index)].dropna()
            n_unique = unique_vals.nunique()

            if n_unique < 2:
                continue  # skip single-value columns

            # For high cardinality: keep top N by entity count, rest → Other
            if n_unique > top_n:
                top_vals = unique_vals.value_counts().head(top_n).index
                col_map = col_map.map(lambda v, tv=top_vals: v if v in tv else "Other")
                was_collapsed = True
            else:
                was_collapsed = False

            # Compute per-segment stats
            seg_data: dict[str, dict] = {}
            for eid in change_series.index:
                seg = col_map.get(eid)
                if seg is None or (isinstance(seg, float) and np.isnan(seg)):
                    seg = "Unknown"
                seg = str(seg)
                if seg not in seg_data:
                    seg_data[seg] = {"total": 0.0, "count": 0, "increases": 0, "decreases": 0}
                val = float(change_series[eid])
                seg_data[seg]["total"] += val
                seg_data[seg]["count"] += 1
                if val > 0:
                    seg_data[seg]["increases"] += 1
                elif val < 0:
                    seg_data[seg]["decreases"] += 1

            segments: dict[str, dict] = {}
            for seg, info in seg_data.items():
                mean_chg = info["total"] / info["count"] if info["count"] > 0 else 0
                contribution = round(info["total"] / total_delta * 100, 1) if total_delta != 0 else 0
                segments[seg] = {
                    "n_entities": info["count"],
                    "mean_change": round(mean_chg, 2),
                    "total_change": round(info["total"], 2),
                    "contribution_pct": contribution,
                    "pct_increased": round(info["increases"] / info["count"] * 100, 1) if info["count"] > 0 else 0,
                    "pct_decreased": round(info["decreases"] / info["count"] * 100, 1) if info["count"] > 0 else 0,
                }

            # Sort by absolute contribution, filter to impactful
            sorted_segs = sorted(segments.items(), key=lambda x: abs(x[1]["contribution_pct"]), reverse=True)
            impactful = [(s, d) for s, d in sorted_segs if abs(d["contribution_pct"]) >= min_contribution_pct]

            if not impactful:
                continue  # skip columns with no impactful segments

            by_column[col] = {
                "n_categories": n_unique,
                "was_collapsed": was_collapsed,
                "segments": dict(impactful),
            }

            # Add to flat key_drivers list
            for seg_name, seg_info in impactful:
                all_drivers.append({
                    "column": col,
                    "value": seg_name,
                    **seg_info,
                })

        # Sort all drivers by absolute contribution
        all_drivers.sort(key=lambda x: abs(x["contribution_pct"]), reverse=True)

        auto_drivers = {
            "total_delta": round(total_delta, 2),
            "n_columns_analyzed": len(cat_cols),
            "n_columns_with_impact": len(by_column),
            "min_contribution_threshold_pct": min_contribution_pct,
            "key_drivers": all_drivers[:20],  # top 20 across all dimensions
            "by_column": by_column,
        }

    data_quality = _nan_report(df[metric], metric)

    result: dict[str, Any] = {
        "analysis_type": "period_comparison",
        "metric": metric,
        "entity_col": entity_col,
        "time_col": time_col,
        "grain": "panel",
        "available_periods": period_labels,
        "aggregation_note": "mean = sum / obs_count (no avg of avg). Entity matching via inner join on entity_col.",
        "data_quality": data_quality,
        "aggregate": aggregate,
        "cohort": cohort,
        "movement": movement,
        "transition_matrix": transition,
        "top_movers": top_movers_result,
    }

    if auto_drivers:
        result["auto_drivers"] = auto_drivers

    return result


def _match_period(user_period: str, periods: list) -> Any:
    """Match a user-supplied period string to actual period values.

    The LLM extraction should return exact period values from the
    time_periods list. This is a fallback for substring/partial matches.
    """
    user_str = str(user_period).strip()
    # Exact match (most common — LLM returns exact value)
    for p in periods:
        if str(p) == user_str:
            return p
    # Substring: user string contained in period or vice versa
    for p in periods:
        p_str = str(p).lower()
        if user_str.lower() in p_str or p_str in user_str.lower():
            return p
    # Try parsing both sides as dates and comparing
    try:
        user_dt = pd.to_datetime(user_str)
        for p in periods:
            if pd.to_datetime(str(p)) == user_dt:
                return p
    except (ValueError, TypeError):
        pass
    # Default: return first period
    return periods[0]
