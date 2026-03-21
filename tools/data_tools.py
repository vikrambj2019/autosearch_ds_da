"""
MCP tools for data profiling, code execution, and column validation.

These run in-process (no subprocess overhead) and provide deterministic,
sandboxed operations that agents call via the Claude SDK MCP protocol.
"""

from __future__ import annotations

import json
import re
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_DF_CACHE: dict[str, tuple[pd.DataFrame, float]] = {}
_CACHE_TTL = 300  # seconds


def _load_df(data_path: str) -> pd.DataFrame:
    """Load a CSV with simple TTL cache so repeated tool calls don't re-read."""
    now = time.time()
    if data_path in _DF_CACHE:
        df, ts = _DF_CACHE[data_path]
        if now - ts < _CACHE_TTL:
            return df

    path = Path(data_path)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    if path.suffix.lower() == ".parquet":
        df = pd.read_parquet(path)
    else:
        df = pd.read_csv(path)

    _DF_CACHE[data_path] = (df, now)
    return df


def _truncate(text: str, max_len: int = 50000) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 40] + f"\n... (truncated, {len(text)} chars total)"


# ---------------------------------------------------------------------------
# 1. profile_data
# ---------------------------------------------------------------------------

def _detect_grain(df: pd.DataFrame) -> dict[str, Any]:
    """Detect the observation grain of a DataFrame.

    Strategy:
      1. Score columns as candidate entity keys (high cardinality, no nulls,
         string/object type, name heuristics like *_ID, *_KEY).
      2. Score columns as candidate time dimensions (date-parseable, limited
         unique values, ordered).
      3. Test uniqueness of (entity, time) to classify grain type.
    """
    n_rows = len(df)
    grain: dict[str, Any] = {
        "entity_col": None,
        "time_col": None,
        "grain_type": "unknown",
        "n_entities": None,
        "n_periods": None,
        "is_balanced": None,
        "aggregation_guidance": {},
    }

    # --- Candidate entity columns ---
    entity_scores: list[tuple[str, float]] = []
    for col in df.columns:
        score = 0.0
        nuniq = df[col].nunique()
        pct_null = df[col].isnull().mean()

        # Must have low nulls
        if pct_null > 0.05:
            continue

        # Cardinality: should be high but < n_rows (otherwise every row is unique → maybe just an ID)
        ratio = nuniq / n_rows if n_rows > 0 else 0
        if ratio < 0.01:
            continue  # too few uniques to be an entity key

        # Name heuristics
        col_upper = col.upper()
        if any(tok in col_upper for tok in ("_ID", "MEMBER", "PATIENT", "PERSON", "USER", "ACCOUNT", "KEY")):
            score += 3.0
        if col_upper.startswith("ID") or col_upper.endswith("ID"):
            score += 1.5

        # Prefer string/object columns for entity keys
        if df[col].dtype == "object":
            score += 1.0

        # Penalize numeric-only values (likely a metric, not an ID)
        if pd.api.types.is_numeric_dtype(df[col]):
            # Check if values look like IDs (integers, no decimals)
            if pd.api.types.is_float_dtype(df[col]):
                score -= 2.0
            else:
                score += 0.5  # integers can be IDs

        # Cardinality score: more unique = more likely entity key
        score += min(ratio * 5, 3.0)

        if score > 0:
            entity_scores.append((col, score))

    entity_scores.sort(key=lambda x: x[1], reverse=True)

    # --- Candidate time columns ---
    time_scores: list[tuple[str, float]] = []
    for col in df.columns:
        score = 0.0
        nuniq = df[col].nunique()

        # Time columns have limited unique values (periods)
        if nuniq > 366 or nuniq < 2:
            continue

        # Name heuristics
        col_upper = col.upper()
        if any(tok in col_upper for tok in ("DATE", "MONTH", "YEAR", "TIME", "PERIOD", "QUARTER", "WEEK")):
            score += 3.0

        # Try parsing as date
        if df[col].dtype == "object":
            sample = df[col].dropna().head(20)
            try:
                parsed = pd.to_datetime(sample, format="mixed")
                if parsed.notna().all():
                    score += 2.0
            except (ValueError, TypeError):
                pass
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            score += 2.5

        if score > 0:
            time_scores.append((col, score))

    time_scores.sort(key=lambda x: x[1], reverse=True)

    # --- Classify grain ---
    entity_col = entity_scores[0][0] if entity_scores else None
    time_col = time_scores[0][0] if time_scores else None

    # Critical check: if entity_col is already unique per row,
    # there's no time dimension — it's cross-sectional, not panel.
    if entity_col and df[entity_col].nunique() == n_rows:
        time_col = None

    if entity_col and time_col:
        n_entities = df[entity_col].nunique()
        n_periods = df[time_col].nunique()

        # Capture actual time period values
        time_values = sorted(df[time_col].dropna().unique(), key=str)
        time_periods_list = [str(v) for v in time_values]

        # Test if (entity, time) is unique → panel
        combo_count = df.groupby([entity_col, time_col]).size()
        is_unique = combo_count.max() == 1
        expected_balanced = n_entities * n_periods
        is_balanced = len(df) == expected_balanced

        if is_unique:
            grain.update({
                "entity_col": entity_col,
                "time_col": time_col,
                "grain_type": "panel",
                "n_entities": int(n_entities),
                "n_periods": int(n_periods),
                "time_periods": time_periods_list,
                "is_balanced": is_balanced,
                "aggregation_guidance": {
                    "per_entity_metric": f"df.groupby('{entity_col}')[metric].mean()  # then .mean() or .median() for overall",
                    "over_time": f"df.groupby('{time_col}')[metric].sum()  # or .mean() depending on question",
                    "entity_count": f"df['{entity_col}'].nunique()  # NOT len(df)",
                    "ml_split": f"Split by '{entity_col}', NOT by row, to avoid data leakage",
                },
            })
        else:
            grain.update({
                "entity_col": entity_col,
                "time_col": time_col,
                "grain_type": "transaction",
                "n_entities": int(n_entities),
                "n_periods": int(n_periods),
                "time_periods": time_periods_list,
                "is_balanced": False,
                "aggregation_guidance": {
                    "per_entity_metric": f"df.groupby('{entity_col}')[metric].sum()  # transactions sum to entity totals",
                    "over_time": f"df.groupby('{time_col}')[metric].sum()",
                    "entity_count": f"df['{entity_col}'].nunique()",
                    "ml_split": f"Split by '{entity_col}', NOT by row",
                },
            })

    elif entity_col and not time_col:
        n_entities = df[entity_col].nunique()
        is_unique = n_entities == n_rows

        if is_unique:
            grain.update({
                "entity_col": entity_col,
                "time_col": None,
                "grain_type": "cross_sectional",
                "n_entities": int(n_entities),
                "n_periods": None,
                "is_balanced": None,
                "aggregation_guidance": {
                    "per_entity_metric": "Each row IS one entity — aggregate directly",
                    "entity_count": f"len(df) or df['{entity_col}'].nunique()  # same thing",
                    "ml_split": "Random train/test split is fine (one row per entity)",
                },
            })
        else:
            grain.update({
                "entity_col": entity_col,
                "time_col": None,
                "grain_type": "entity_multi_row",
                "n_entities": int(n_entities),
                "n_periods": None,
                "is_balanced": False,
                "aggregation_guidance": {
                    "per_entity_metric": f"df.groupby('{entity_col}')[metric].agg(...)  # aggregate first",
                    "entity_count": f"df['{entity_col}'].nunique()  # NOT len(df)",
                    "ml_split": f"Split by '{entity_col}', NOT by row",
                },
            })

    elif time_col and not entity_col:
        grain.update({
            "entity_col": None,
            "time_col": time_col,
            "grain_type": "time_series",
            "n_entities": None,
            "n_periods": int(df[time_col].nunique()),
            "is_balanced": None,
            "aggregation_guidance": {
                "over_time": f"df.groupby('{time_col}')[metric].sum()",
            },
        })

    return grain


def _column_profile(df: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-column profile: dtype, nulls, cardinality, basic stats."""
    profiles = []
    for col in df.columns:
        series = df[col]
        info: dict[str, Any] = {
            "name": col,
            "dtype": str(series.dtype),
            "n_null": int(series.isnull().sum()),
            "pct_null": round(float(series.isnull().mean()) * 100, 1),
            "n_unique": int(series.nunique()),
        }

        if pd.api.types.is_numeric_dtype(series):
            desc = series.describe()
            info["min"] = round(float(desc.get("min", 0)), 2)
            info["max"] = round(float(desc.get("max", 0)), 2)
            info["mean"] = round(float(desc.get("mean", 0)), 2)
            info["std"] = round(float(desc.get("std", 0)), 2)
            info["median"] = round(float(series.median()), 2)
        else:
            top = series.value_counts().head(5)
            info["top_values"] = {str(k): int(v) for k, v in top.items()}

        profiles.append(info)
    return profiles


def _data_warnings(df: pd.DataFrame, grain: dict) -> list[str]:
    """Flag common data quality issues."""
    warnings = []
    for col in df.columns:
        pct_null = df[col].isnull().mean()
        if pct_null > 0.30:
            warnings.append(f"{col}: {pct_null:.0%} null values")

    if grain["grain_type"] == "panel" and not grain.get("is_balanced"):
        warnings.append(
            f"Unbalanced panel: {grain['n_entities']} entities x {grain['n_periods']} periods "
            f"= {grain['n_entities'] * grain['n_periods']} expected but {len(df)} actual rows"
        )

    # Check for constant columns
    for col in df.columns:
        if df[col].nunique() <= 1:
            warnings.append(f"{col}: constant column (only 1 unique value)")

    return warnings


async def profile_data_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Profile a dataset: schema, grain detection, stats, sample rows."""
    data_path = args["data_path"]
    sample_rows = args.get("sample_rows", 5)

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    grain = _detect_grain(df)
    columns = _column_profile(df)
    warnings = _data_warnings(df, grain)

    # For wide datasets, limit sample columns to keep output manageable
    sample_df = df.head(int(sample_rows))
    if len(sample_df.columns) > 20:
        sample_df = sample_df.iloc[:, :20]
        sample_note = f"(showing first 20 of {len(df.columns)} columns)"
    else:
        sample_note = None
    sample = sample_df.to_string(index=False)

    # For wide datasets, cap detailed column profiles and summarize the rest
    if len(columns) > 50:
        detailed_columns = columns[:50]
        remaining = [c["name"] for c in columns[50:]]
        columns_output = detailed_columns
        columns_note = f"Showing detailed profiles for first 50 columns. Remaining {len(remaining)}: {remaining}"
    else:
        columns_output = columns
        columns_note = None

    result: dict[str, Any] = {
        "data_path": data_path,
        "shape": list(df.shape),
        "memory_mb": round(df.memory_usage(deep=True).sum() / 1024 / 1024, 2),
        "grain": grain,
        "columns": columns_output,
        "warnings": warnings,
        "sample": sample,
    }
    if sample_note:
        result["sample_note"] = sample_note
    if columns_note:
        result["columns_note"] = columns_note

    text = json.dumps(result, indent=2, default=str, ensure_ascii=True)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


# ---------------------------------------------------------------------------
# 2. run_code
# ---------------------------------------------------------------------------

# Allowed modules in the sandbox
_SAFE_BUILTINS = {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict,
    "enumerate": enumerate, "filter": filter, "float": float, "format": format,
    "frozenset": frozenset, "int": int, "isinstance": isinstance,
    "len": len, "list": list, "map": map, "max": max, "min": min,
    "print": print, "range": range, "reversed": reversed, "round": round,
    "set": set, "slice": slice, "sorted": sorted, "str": str, "sum": sum,
    "tuple": tuple, "type": type, "zip": zip,
    "True": True, "False": False, "None": None,
}

# Blocked patterns in code
_BLOCKED_PATTERNS = [
    r"\bimport\s+os\b",
    r"\bimport\s+sys\b",
    r"\bimport\s+subprocess\b",
    r"\bimport\s+shutil\b",
    r"\b__import__\b",
    r"\bexec\s*\(",
    r"\beval\s*\(",
    r"\bopen\s*\(",
    r"\bos\.",
    r"\bsubprocess\.",
    r"\bsys\.",
]


def _validate_code(code: str) -> str | None:
    """Return error message if code contains blocked patterns, else None."""
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, code):
            return f"Blocked pattern detected: {pattern}"
    return None


def _format_result(result: Any, fmt: str) -> tuple[str, list[int] | None]:
    """Format execution result. Returns (text, shape_or_none)."""
    shape = None

    if isinstance(result, pd.DataFrame):
        shape = list(result.shape)
        if len(result) > 500:
            result = result.head(500)
        if fmt == "markdown":
            text = result.to_markdown(index=True)
        elif fmt == "json":
            text = result.to_json(orient="records", indent=2, default_handler=str)
        else:  # table (default)
            text = result.to_string()
    elif isinstance(result, pd.Series):
        shape = [len(result)]
        if len(result) > 500:
            result = result.head(500)
        if fmt == "markdown":
            text = result.to_markdown()
        elif fmt == "json":
            text = result.to_json(indent=2, default_handler=str)
        else:
            text = result.to_string()
    elif isinstance(result, (np.integer, np.floating)):
        text = str(result.item())
    elif isinstance(result, np.ndarray):
        shape = list(result.shape)
        text = str(result)
    else:
        text = str(result)

    return text, shape


async def run_code_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Execute pandas code in a sandboxed namespace.

    The CSV is pre-loaded as `df`. Available: pd, np, stats (scipy.stats).
    """
    code = args["code"]
    data_path = args["data_path"]
    return_format = args.get("return_format", "table")

    # Validate code safety
    violation = _validate_code(code)
    if violation:
        return {
            "content": [{"type": "text", "text": f"BLOCKED: {violation}"}],
            "is_error": True,
        }

    # Load data
    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    # Build sandboxed namespace
    namespace: dict[str, Any] = {
        "__builtins__": _SAFE_BUILTINS,
        "df": df,
        "pd": pd,
        "np": np,
        "stats": stats,
    }

    t0 = time.time()
    try:
        # If code is a single expression, eval it; otherwise exec and look for `result`
        try:
            result = eval(code, namespace)  # noqa: S307 — sandboxed namespace
        except SyntaxError:
            exec(code, namespace)  # noqa: S102 — sandboxed namespace
            result = namespace.get("result", "(no `result` variable set — assign your output to `result`)")

        elapsed_ms = int((time.time() - t0) * 1000)
        text, shape = _format_result(result, return_format)

        output = {
            "success": True,
            "result": text,
            "result_shape": shape,
            "execution_time_ms": elapsed_ms,
        }
    except Exception:
        elapsed_ms = int((time.time() - t0) * 1000)
        output = {
            "success": False,
            "error": traceback.format_exc(),
            "execution_time_ms": elapsed_ms,
        }

    return {"content": [{"type": "text", "text": _truncate(json.dumps(output, indent=2, default=str))}]}


# ---------------------------------------------------------------------------
# 3. validate_cols
# ---------------------------------------------------------------------------

def _fuzzy_match_column(user_ref: str, actual_cols: list[str]) -> tuple[str | None, float]:
    """Match a user column reference to actual column names.

    Returns (best_match, confidence 0-1).
    """
    user_ref_upper = user_ref.strip().upper().replace(" ", "_")
    user_tokens = set(user_ref_upper.replace("_", " ").split())

    best_match = None
    best_score = 0.0

    for col in actual_cols:
        col_upper = col.upper()
        col_tokens = set(col_upper.replace("_", " ").split())

        # Exact match
        if user_ref_upper == col_upper:
            return col, 1.0

        score = 0.0

        # Substring match: user_ref is contained in column name
        if user_ref_upper in col_upper:
            score = max(score, 0.8)

        # Column name is contained in user_ref
        if col_upper in user_ref_upper:
            score = max(score, 0.7)

        # Token overlap
        if user_tokens and col_tokens:
            overlap = len(user_tokens & col_tokens)
            token_score = overlap / max(len(user_tokens), len(col_tokens))
            score = max(score, token_score * 0.9)

        # Abbreviation handling
        abbrevs = {
            "AVG": "AVG", "ER": "ER", "COST": "COST", "TOTAL": "TOTAL",
            "MONTHLY": "MONTHLY", "INCOME": "INCOME", "ACTIVE": "ACTIVE",
            "COUNTY": "COUNTY", "CITY": "CITY", "STATE": "STATE",
            "SEX": "SEX", "GENDER": "SEX", "MEMBER": "MEMBER",
            "DATE": "DATE", "MONTH": "MONTH", "YEAR": "YEAR",
        }
        for abbr, expanded in abbrevs.items():
            if abbr in user_ref_upper and expanded in col_upper:
                score = max(score, 0.6)

        if score > best_score:
            best_score = score
            best_match = col

    return best_match, best_score


async def validate_cols_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Fuzzy-match user column references to actual column names."""
    data_path = args["data_path"]
    user_columns_str = args["user_columns"]

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    actual_cols = list(df.columns)
    user_refs = [c.strip() for c in user_columns_str.split(",") if c.strip()]

    matches = {}
    unmatched = []
    suggestions = {}

    for ref in user_refs:
        match, confidence = _fuzzy_match_column(ref, actual_cols)
        if match and confidence >= 0.5:
            matches[ref] = {"column": match, "confidence": round(confidence, 2)}
        else:
            unmatched.append(ref)
            # Find top 3 closest for suggestions
            scored = [(col, _fuzzy_match_column(ref, [col])[1]) for col in actual_cols]
            scored.sort(key=lambda x: x[1], reverse=True)
            suggestions[ref] = [col for col, _ in scored[:3]]

    result = {
        "matches": matches,
        "unmatched": unmatched,
        "suggestions": suggestions,
        "all_columns": actual_cols,
    }

    return {"content": [{"type": "text", "text": json.dumps(result, indent=2)}]}


# ---------------------------------------------------------------------------
# Grain context builder (for prompt injection)
# ---------------------------------------------------------------------------


def build_grain_context(data_path: str) -> str:
    """Profile a dataset and return a human-readable grain context string.

    This is called BEFORE the agent loop so that every agent prompt
    contains grounded facts about the data — not hallucinated ones.
    """
    try:
        df = _load_df(data_path)
    except FileNotFoundError:
        return f"Data file not found: {data_path}"

    grain = _detect_grain(df)
    warnings = _data_warnings(df, grain)

    # Build the context string
    lines = [
        f"Dataset: {data_path}",
        f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns",
        f"Grain type: {grain['grain_type']}",
    ]

    if grain["entity_col"]:
        lines.append(f"Entity column: {grain['entity_col']} ({grain['n_entities']:,} unique entities)")
    if grain["time_col"]:
        lines.append(f"Time column: {grain['time_col']} ({grain['n_periods']} periods)")
        periods = grain.get("time_periods", [])
        if periods:
            lines.append(f"Time periods: {', '.join(str(p) for p in periods)}")
    if grain.get("is_balanced") is not None:
        lines.append(f"Balanced panel: {grain['is_balanced']}")

    # Column summary
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()
    lines.append(f"Numeric columns ({len(numeric_cols)}): {', '.join(numeric_cols[:20])}")
    if cat_cols:
        lines.append(f"Categorical columns ({len(cat_cols)}): {', '.join(cat_cols[:20])}")

    if warnings:
        lines.append(f"Data warnings: {'; '.join(warnings)}")

    # Aggregation guidance
    guidance = grain.get("aggregation_guidance", {})
    if guidance:
        lines.append("\nAggregation rules:")
        for key, val in guidance.items():
            lines.append(f"  - {key}: {val}")

    lines.append(
        "\nIMPORTANT: Only reference facts from this profile. "
        "Do NOT invent months, time periods, or row counts that are not listed above."
    )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# 4. Deterministic analytics tools
# ---------------------------------------------------------------------------

async def distribution_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Distribution of a metric by category — deterministic, grain-aware."""
    from .analytics import distribution_by_category

    data_path = args["data_path"]
    metric = args["metric"]
    category = args["category"]
    top_n = args.get("top_n", 10)

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    result = distribution_by_category(df, metric, category, top_n=top_n)
    text = json.dumps(result, indent=2, default=str)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


async def trend_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Trend of a metric over time — deterministic, grain-aware."""
    from .analytics import trend_over_time

    data_path = args["data_path"]
    metric = args["metric"]
    stratify_by = args.get("stratify_by")
    top_n = args.get("top_n", 10)

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    result = trend_over_time(df, metric, stratify_by=stratify_by, top_n=top_n)
    text = json.dumps(result, indent=2, default=str)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


async def comparison_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Statistical comparison of a metric between groups — deterministic, grain-aware."""
    from .analytics import group_comparison

    data_path = args["data_path"]
    metric = args["metric"]
    group_col = args["group_col"]
    group_a = args.get("group_a")
    group_b = args.get("group_b")

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    result = group_comparison(df, metric, group_col, group_a=group_a, group_b=group_b)
    text = json.dumps(result, indent=2, default=str)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


async def correlation_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Correlation between two metrics — deterministic, grain-aware."""
    from .analytics import correlation_analysis

    data_path = args["data_path"]
    metric_a = args["metric_a"]
    metric_b = args["metric_b"]

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    result = correlation_analysis(df, metric_a, metric_b)
    text = json.dumps(result, indent=2, default=str)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


async def summary_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Summary statistics for a metric — deterministic, grain-aware."""
    from .analytics import summary_stats

    data_path = args["data_path"]
    metric = args["metric"]

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    result = summary_stats(df, metric)
    text = json.dumps(result, indent=2, default=str)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


async def period_comparison_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Period-over-period entity-level comparison — deterministic, panel-only."""
    from .analytics import period_comparison

    data_path = args["data_path"]
    metric = args["metric"]
    period_a = args.get("period_a")
    period_b = args.get("period_b")
    stratify_by = args.get("stratify_by")
    top_n = args.get("top_n", 10)
    top_movers = args.get("top_movers", 10)

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    result = period_comparison(
        df, metric,
        period_a=period_a, period_b=period_b,
        stratify_by=stratify_by, top_n=top_n, top_movers=top_movers,
    )
    text = json.dumps(result, indent=2, default=str)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


async def entity_counts_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Count unique entities, optionally by group — deterministic, grain-aware."""
    from .analytics import entity_counts

    data_path = args["data_path"]
    group_col = args.get("group_col")
    top_n = args.get("top_n", 10)

    try:
        df = _load_df(data_path)
    except FileNotFoundError as e:
        return {"content": [{"type": "text", "text": f"ERROR: {e}"}], "is_error": True}

    result = entity_counts(df, group_col=group_col, top_n=top_n)
    text = json.dumps(result, indent=2, default=str)
    return {"content": [{"type": "text", "text": _truncate(text)}]}


# ---------------------------------------------------------------------------
# Convenience: standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import asyncio

    async def _test():
        # Test profile_data
        print("=" * 60)
        print("TEST: profile_data (panel)")
        print("=" * 60)
        result = await profile_data_handler({
            "data_path": "/Users/vikrambandugula/Documents/segmentation/MLModels/agent_test_vb/data/ds_data/pannel_data.csv",
            "sample_rows": 3,
        })
        profile = json.loads(result["content"][0]["text"])
        print(f"Shape: {profile['shape']}")
        print(f"Grain: {json.dumps(profile['grain'], indent=2)}")
        print(f"Warnings: {profile['warnings']}")
        print()

        # Test profile_data (cross-sectional)
        print("=" * 60)
        print("TEST: profile_data (cross-sectional)")
        print("=" * 60)
        result = await profile_data_handler({
            "data_path": "/Users/vikrambandugula/Documents/segmentation/MLModels/agent_test_vb/data/ds_data/raw_data.csv",
            "sample_rows": 2,
        })
        profile = json.loads(result["content"][0]["text"])
        print(f"Shape: {profile['shape']}")
        print(f"Grain: {json.dumps(profile['grain'], indent=2)}")
        print()

        # Test run_code
        print("=" * 60)
        print("TEST: run_code (90th percentile by county)")
        print("=" * 60)
        result = await run_code_handler({
            "code": "df.groupby('COUNTY_LATEST')['MONTHLY_TOTAL_COST'].quantile(0.9).round(2).sort_values(ascending=False).head(10)",
            "data_path": "/Users/vikrambandugula/Documents/segmentation/MLModels/agent_test_vb/data/ds_data/pannel_data.csv",
            "return_format": "table",
        })
        output = json.loads(result["content"][0]["text"])
        print(f"Success: {output['success']}")
        print(f"Time: {output['execution_time_ms']}ms")
        print(output.get("result", output.get("error", "")))
        print()

        # Test run_code — grain-aware member-level aggregation
        print("=" * 60)
        print("TEST: run_code (avg cost per member, grain-aware)")
        print("=" * 60)
        result = await run_code_handler({
            "code": "df.groupby('FEATURE_STORE_MEMBER_ID')['MONTHLY_TOTAL_COST'].mean().describe().round(2)",
            "data_path": "/Users/vikrambandugula/Documents/segmentation/MLModels/agent_test_vb/data/ds_data/pannel_data.csv",
            "return_format": "table",
        })
        output = json.loads(result["content"][0]["text"])
        print(f"Success: {output['success']}")
        print(output.get("result", output.get("error", "")))
        print()

        # Test run_code — blocked pattern
        print("=" * 60)
        print("TEST: run_code (blocked pattern)")
        print("=" * 60)
        result = await run_code_handler({
            "code": "import os; os.listdir('/')",
            "data_path": "/Users/vikrambandugula/Documents/segmentation/MLModels/agent_test_vb/data/ds_data/pannel_data.csv",
            "return_format": "table",
        })
        print(result["content"][0]["text"])
        print()

        # Test validate_cols
        print("=" * 60)
        print("TEST: validate_cols")
        print("=" * 60)
        result = await validate_cols_handler({
            "data_path": "/Users/vikrambandugula/Documents/segmentation/MLModels/agent_test_vb/data/ds_data/pannel_data.csv",
            "user_columns": "county, ER_COST, total cost, income, active status, member ID",
        })
        print(result["content"][0]["text"])

    asyncio.run(_test())
