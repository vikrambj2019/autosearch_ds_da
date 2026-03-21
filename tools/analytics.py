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
) -> dict[str, Any]:
    """Distribution of a metric grouped by a categorical variable.

    mean = sum(metric) / count(observations) per group — no avg of avg.
    Entity counts use nunique(entity_col) for panel data.
    """
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
) -> dict[str, Any]:
    """Statistical comparison of a metric between groups.

    Uses raw observations — mean = sum / obs_count.
    2 groups: Welch's t-test + Mann-Whitney + Cohen's d
    3+ groups: ANOVA + Kruskal-Wallis
    """
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
) -> dict[str, Any]:
    """Correlation between two numeric columns.

    Uses raw observations. For panel data, also reports entity count.
    """
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)
    dq_a = _nan_report(df[metric_a], metric_a)
    dq_b = _nan_report(df[metric_b], metric_b)

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
        "grain": "panel" if is_panel else "cross_sectional",
        "aggregation_note": "computed on raw observations (one row per member-month)",
        "n_observations": n_obs,
        "n_entities": n_entities,
        "data_quality": [dq_a, dq_b],
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
            "summary": f"{strength} {sign} correlation (r={pearson_r:.3f}, p={pearson_p:.4f})",
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
) -> dict[str, Any]:
    """Summary statistics for a single metric.

    Uses raw observations. mean = sum / count (no avg of avg).
    """
    if entity_col is None and time_col is None:
        entity_col, time_col = _detect_entity_time(df)

    is_panel = _is_panel(df, entity_col, time_col)
    data_quality = _nan_report(df[metric], metric)

    vals = df[metric].dropna()
    n_obs = len(vals)
    total_sum = float(vals.sum())
    mean = total_sum / n_obs if n_obs > 0 else 0
    n_entities = int(df[entity_col].nunique()) if entity_col else n_obs

    desc = vals.describe()
    ci = _confidence_interval(vals)

    return {
        "analysis_type": "summary_stats",
        "metric": metric,
        "grain": "panel" if is_panel else "cross_sectional",
        "aggregation_note": "mean = total_sum / obs_count on raw observations (no avg of avg)",
        "computed_on": f"raw observations ({n_obs} obs from {n_entities} entities)" if is_panel else f"raw rows ({n_obs})",
        "n_entities": n_entities,
        "n_observations": n_obs,
        "data_quality": data_quality,
        "statistics": {
            "total_sum": round(total_sum, 2),
            "mean": round(mean, 2),
            "std": round(float(desc["std"]), 2),
            "min": round(float(desc["min"]), 2),
            "p25": round(float(desc["25%"]), 2),
            "median": round(float(desc["50%"]), 2),
            "p75": round(float(desc["75%"]), 2),
            "iqr": round(float(desc["75%"] - desc["25%"]), 2),
            "p90": round(float(vals.quantile(0.90)), 2),
            "p95": round(float(vals.quantile(0.95)), 2),
            "p99": round(float(vals.quantile(0.99)), 2),
            "max": round(float(desc["max"]), 2),
            "ci_95": ci,
        },
        "shape": {
            "skewness": round(float(vals.skew()), 3),
            "zero_pct": round(float((vals == 0).mean()) * 100, 1),
            "mean_to_median_ratio": round(mean / float(desc["50%"]), 2) if desc["50%"] > 0 else None,
        },
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
# 7. period_comparison (entity-level diff between two time periods)
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
