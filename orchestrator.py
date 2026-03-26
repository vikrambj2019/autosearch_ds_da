"""
Orchestrator — 2-pass analytics engine.

Architecture:
  Pass 1: Python calls the right deterministic tool for the main question.
  Pass 2: Python picks 2 depth analyses based on Pass 1 results, runs them.
  Narrate: Single LLM call turns all tool outputs into a report.

The LLM NEVER computes numbers. Every number comes from deterministic Python.
The LLM ONLY writes English narrative from pre-computed JSON results.

Usage:
    python3 orchestrator.py --data Data/pannel_data.csv "What is the distribution of MONTHLY_TOTAL_COST by CURRENTLY_ACTIVE?"
    python3 orchestrator.py --data Data/pannel_data.csv "Is there a significant difference in cost between male and female?"
    python3 orchestrator.py --data Data/raw_data.csv "Build me a classification model predicting TARGET_HIGH_COST_FLAG"
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from datetime import datetime
from pathlib import Path

import os

import anthropic
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

_anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
_MODEL = "claude-sonnet-4-6"

from agents.definitions import (
    discover_analytics_skills,
    route_question,
)
from tools.data_tools import _load_df, _detect_grain, build_grain_context
from tools.analytics import (
    distribution_by_category,
    trend_over_time,
    group_comparison,
    correlation_analysis,
    summary_stats,
    normality_test,
    entity_lookup,
    entity_counts,
    price_volume_mix,
    period_comparison,
)


# ── Helpers ──────────────────────────────────────────────────────────────────


def _collect_text(response: anthropic.types.Message) -> str:
    return "\n".join(
        block.text for block in response.content if block.type == "text"
    )


# ── Parameter Extraction (LLM-based) ─────────────────────────────────────────

_PARAM_EXTRACTION_PROMPT = """\
You are a parameter extractor for a data analytics system. Given a user question and dataset metadata, extract the structured parameters needed to run the analysis.

## Available analysis types
- **distribution**: Breakdown/distribution of a numeric metric by a categorical column (e.g., "cost by gender", "spending breakdown by county")
- **trend**: Time series trend of a metric (e.g., "cost over time", "monthly trend")
- **comparison**: Statistical comparison between groups (e.g., "difference in cost between male and female", "is X higher for group A vs B")
- **correlation**: Relationship between two numeric metrics (e.g., "correlation between cost and income")
- **summary**: Descriptive statistics for a metric (e.g., "summary of cost", "percentiles", "p90")
- **normality**: Test whether a column follows a normal distribution (e.g., "is X normally distributed?", "normality test on column Y", "does the distribution adhere to normal")
- **lookup**: Find which entity has the highest/lowest value (e.g., "which country has the highest happiness score?", "which field has the most graduates?", "find the site with the highest value"). Use when the question asks "which X has the highest/lowest Y".
- **entity_counts**: Count of unique entities (e.g., "how many members", "count of patients")
- **pvm**: Price-Volume-Mix decomposition of revenue or margin change between two periods (e.g., "why did revenue change?", "price volume mix analysis", "what drove the margin change?", "decompose the revenue change into price, volume, and mix"). Use when the question asks about decomposing revenue/margin changes into price, volume, and mix components. Requires qty and revenue columns.
- **period_comparison**: Entity-level comparison between two time periods (e.g., "compare July vs September", "what changed between months", "which members drove the cost increase"). Only for panel data with a time column.

## Rules
- **metric**: Must be an EXACT column name from the provided list. Pick the numeric column the user is asking about. If ambiguous, prefer the most general cost/amount column.
- **category**: Must be an EXACT column name from the provided list, or null. This is the grouping/segmentation column. Do NOT pick the entity ID or time column.
- **metric_b**: Only for correlation analysis — the second numeric column. Must be exact column name or null.
- **period_a**: Only for period_comparison — must be an EXACT value from the provided `time_periods` list. Match the user's reference (e.g., "July" → "2025-07-01") to the closest period in the list. Null if user doesn't specify.
- **period_b**: Only for period_comparison — must be an EXACT value from the provided `time_periods` list. Null if user doesn't specify.
- **stratify_by**: Only for period_comparison or trend — categorical column to segment the analysis. Null if not specified.
- **group_by**: Column(s) to compute the metric SEPARATELY for each group. Can be a single column name (string) or a list of column names for multi-level grouping. Use when the question asks "for each class", "per month", "by gender and class", "for male vs female passengers in each class", "as a function of X", "by income level". Must be EXACT column name(s) or null. Works with both categorical AND numeric columns — numeric columns are auto-binned into quartiles. Different from category: group_by runs the FULL analysis separately per group, while category is the breakdown dimension of a single analysis.
- **filters**: List of filter conditions to apply BEFORE analysis. Use when the question restricts the data subset: "for female passengers", "who survived", "in first class", "with a fare greater than X", "male passengers who survived and were in first class". IMPORTANT: extract ALL conditions as separate filters — multiple filters are AND-combined. Each filter: {"column": "exact_col_name", "op": "==|!=|>|>=|<|<=|in|not_in|contains", "value": ...}. Null if no filtering needed.
- **lookup_mode**: Only for lookup analysis — "max" (default) or "min". Use "max" for "highest", "most", "largest", "best". Use "min" for "lowest", "least", "smallest", "worst".
- **qty_col**: Only for pvm analysis — the column containing quantities. Must be exact column name or null (auto-detected from "qty", "quantity", "units", "volume").
- **revenue_col**: Only for pvm analysis — the column containing revenue. Must be exact column name or null (auto-detected from "rev", "revenue", "sales", "amount").
- **product_col**: Only for pvm analysis — the column containing product/SKU identifiers. Must be exact column name or null (auto-detected from "product_sku", "sku", "product").
- **customer_col**: Only for pvm analysis — the column containing customer IDs. Must be exact column name or null.
- **cost_col**: Only for pvm analysis — the column containing costs. Null if not available.
- **margin_col**: Only for pvm analysis — the column containing margins. Null if not available.
- **analysis_type**: One of the 10 types above. Use the skill_name hint if provided (diagnostic → comparison or correlation). Use period_comparison when the user asks to compare between specific time periods, or asks "what changed" between months. Use **normality** when the user asks if data is "normally distributed", "follows a normal distribution", or "adheres to normal". Use **lookup** when the question asks "which X has the highest/lowest Y". Use **pvm** when the question asks about price/volume/mix decomposition or why revenue/margin changed.
- If the user mentions values like "male/female" or "active/inactive", find which column contains those values.
- Return ONLY valid JSON, no markdown fences, no explanation.

## Examples

Q: "What is the average age of male passengers in each passenger class?"
→ {"analysis_type": "summary", "metric": "Age", "filters": [{"column": "Sex", "op": "==", "value": "male"}], "group_by": "Pclass"}

Q: "Calculate the correlation between age and fare for passengers who survived and were in first class"
→ {"analysis_type": "correlation", "metric": "Age", "metric_b": "Fare", "filters": [{"column": "Survived", "op": "==", "value": 1}, {"column": "Pclass", "op": "==", "value": 1}]}

Q: "Which country has the highest happiness score?"
→ {"analysis_type": "lookup", "metric": "Happiness Score", "category": "Country", "lookup_mode": "max"}

Q: "Why did revenue change between January and February? Price volume mix analysis"
→ {"analysis_type": "pvm", "metric": "rev", "revenue_col": "rev", "qty_col": "qty", "product_col": "product_sku", "customer_col": "customer_id", "cost_col": "cost", "margin_col": "margin", "period_a": "2025-01-01", "period_b": "2025-02-01"}
"""


async def _extract_params(
    question: str,
    skill_name: str | None,
    columns: list[str],
    grain: dict,
) -> dict:
    """Extract tool parameters from the question using a single LLM call.

    Returns dict with: analysis_type, metric, category, metric_b.
    Falls back to simple keyword matching if the LLM call fails.
    """
    entity_col = grain.get("entity_col")
    time_col = grain.get("time_col")

    # Build column metadata for the LLM
    numeric_cols = [c for c in columns if any(k in c.upper() for k in ("COST", "AMOUNT", "TOTAL", "INCOME", "AVG"))]
    cat_candidates = [
        c for c in columns
        if c != entity_col and c != time_col
        and not any(k in c.upper() for k in ("COST", "AMOUNT", "TOTAL", "INCOME", "AVG"))
    ]

    # Include categorical values so LLM can map "male/female" → SEX, etc.
    cat_value_info = {}
    cat_value_map = grain.get("_cat_values", {})
    for col in cat_candidates:
        vals = cat_value_map.get(col, [])
        if vals:
            cat_value_info[col] = [str(v) for v in vals[:15]]

    # Include actual time period values so LLM can resolve "July" → "2025-07-01"
    time_periods = grain.get("time_periods", [])

    user_msg = json.dumps({
        "question": question,
        "skill_name_hint": skill_name,
        "numeric_columns": numeric_cols,
        "categorical_columns": cat_candidates,
        "categorical_values": cat_value_info,
        "entity_column": entity_col,
        "time_column": time_col,
        "time_periods": time_periods,
        "all_columns": columns,
    }, indent=2)

    try:
        response = _anthropic_client.messages.create(
            model=_MODEL,
            max_tokens=1024,
            system=_PARAM_EXTRACTION_PROMPT,
            messages=[{"role": "user", "content": user_msg}],
        )
        text = _collect_text(response)
        if not text.strip():
            raise ValueError("LLM returned empty response")
        # Parse JSON — strip markdown fences if present
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            text = text.rsplit("```", 1)[0]
        parsed = json.loads(text)

        # Validate: all returned columns must actually exist
        result = {}
        result["analysis_type"] = parsed.get("analysis_type", "distribution")
        valid_types = ("distribution", "trend", "comparison", "correlation", "summary", "normality", "lookup", "entity_counts", "pvm", "period_comparison")
        if result["analysis_type"] not in valid_types:
            result["analysis_type"] = "distribution"

        # lookup fields
        lm = parsed.get("lookup_mode", "max")
        result["lookup_mode"] = lm if lm in ("max", "min") else "max"

        # pvm fields
        for pvm_field in ("qty_col", "revenue_col", "product_col", "customer_col", "cost_col", "margin_col"):
            val = parsed.get(pvm_field)
            result[pvm_field] = val if val in columns else None

        m = parsed.get("metric")
        result["metric"] = m if m in columns else (numeric_cols[0] if numeric_cols else columns[0])

        c = parsed.get("category")
        result["category"] = c if c in columns else None

        mb = parsed.get("metric_b")
        result["metric_b"] = mb if mb in columns else None

        # period_comparison fields
        result["period_a"] = parsed.get("period_a")
        result["period_b"] = parsed.get("period_b")
        sb = parsed.get("stratify_by")
        result["stratify_by"] = sb if sb in columns else None

        # group_by: validate column(s) exist — can be string or list
        gb = parsed.get("group_by")
        if isinstance(gb, list):
            gb = [col for col in gb if col in columns]
            result["group_by"] = gb if gb else None
        else:
            result["group_by"] = gb if gb in columns else None

        # filters: validate each filter's column exists
        raw_filters = parsed.get("filters")
        if raw_filters and isinstance(raw_filters, list):
            valid_filters = [f for f in raw_filters if isinstance(f, dict) and f.get("column") in columns]
            result["filters"] = valid_filters if valid_filters else None
        else:
            result["filters"] = None

        return result

    except Exception as e:
        print(f"  [param extraction] LLM call failed ({e}), using keyword fallback")
        return _extract_params_fallback(question, skill_name, columns, grain)


def _extract_params_fallback(
    question: str,
    skill_name: str | None,
    columns: list[str],
    grain: dict,
) -> dict:
    """Simple keyword fallback if the LLM extraction fails."""
    q_lower = question.lower()
    entity_col = grain.get("entity_col")
    time_col = grain.get("time_col")
    numeric_cols = [c for c in columns if any(k in c.upper() for k in ("COST", "AMOUNT", "TOTAL", "INCOME", "AVG"))]
    default_metric = numeric_cols[0] if numeric_cols else columns[0]

    # Metric: exact column name match
    metric = None
    for col in columns:
        if col.lower() in q_lower or col.replace("_", " ").lower() in q_lower:
            if col in numeric_cols:
                metric = col
                break
    if not metric:
        metric = default_metric

    # Category: exact column name match
    category = None
    cat_candidates = [
        c for c in columns
        if c != metric and c != entity_col and c != time_col and c not in numeric_cols
    ]
    for col in cat_candidates:
        if col.lower() in q_lower or col.replace("_", " ").lower() in q_lower:
            category = col
            break

    # Analysis type: keyword matching
    if skill_name == "diagnostic":
        analysis_type = "correlation" if "correlat" in q_lower else "comparison"
    elif any(w in q_lower for w in ("price volume mix", "pvm", "price effect", "volume effect", "mix effect", "decompose revenue", "decompose margin")):
        analysis_type = "pvm"
    elif any(w in q_lower for w in ("normally distributed", "normal distribution", "normality", "adheres to normal")):
        analysis_type = "normality"
    elif any(w in q_lower for w in ("which", "highest", "lowest", "most", "least")) and any(w in q_lower for w in ("has the", "with the", "find the")):
        analysis_type = "lookup"
    elif any(w in q_lower for w in ("distribution", "breakdown")):
        analysis_type = "distribution"
    elif any(w in q_lower for w in ("trend", "over time")):
        analysis_type = "trend"
    elif any(w in q_lower for w in ("how many", "count")):
        analysis_type = "entity_counts"
    elif any(w in q_lower for w in ("correlat", "relationship")):
        analysis_type = "correlation"
    elif any(w in q_lower for w in ("compare", "difference", "significant", "vs")):
        # Check if comparing periods vs groups
        if any(w in q_lower for w in ("month", "period", "july", "august", "september", "jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")):
            analysis_type = "period_comparison"
        else:
            analysis_type = "comparison"
    elif any(w in q_lower for w in ("changed", "what happened", "between months")):
        analysis_type = "period_comparison"
    else:
        analysis_type = "distribution"

    # metric_b for correlation
    metric_b = None
    if analysis_type == "correlation":
        for col in numeric_cols:
            if col != metric and (col.lower() in q_lower or col.replace("_", " ").lower() in q_lower):
                metric_b = col
                break
        if not metric_b and len(numeric_cols) > 1:
            metric_b = [c for c in numeric_cols if c != metric][0]

    return {
        "analysis_type": analysis_type,
        "metric": metric,
        "category": category,
        "metric_b": metric_b,
        "period_a": None,
        "period_b": None,
        "stratify_by": None,
        "group_by": None,
        "filters": None,
        "lookup_mode": "max",
        "qty_col": None,
        "revenue_col": None,
        "product_col": None,
        "customer_col": None,
        "cost_col": None,
        "margin_col": None,
    }


# ── Pass 1: Primary Analysis ────────────────────────────────────────────────


def _run_primary(params: dict, df, verbose: bool = True) -> dict:
    """Run the primary deterministic analysis."""
    analysis_type = params["analysis_type"]
    metric = params["metric"]
    category = params.get("category")
    filters = params.get("filters")
    group_by = params.get("group_by")

    if verbose:
        extra = []
        if category:
            extra.append(f"cat={category}")
        if group_by:
            extra.append(f"group_by={group_by}")
        if filters:
            extra.append(f"filters={len(filters)}")
        suffix = f", {', '.join(extra)}" if extra else ""
        print(f"\nPass 1 — {analysis_type}({metric}{suffix})")

    if analysis_type == "distribution":
        if not category:
            return summary_stats(df, metric, filters=filters, group_by=group_by)
        return distribution_by_category(df, metric, category, filters=filters)

    elif analysis_type == "trend":
        return trend_over_time(df, metric, stratify_by=category)

    elif analysis_type == "comparison":
        if not category:
            return {"error": "No group column found for comparison"}
        return group_comparison(df, metric, category, filters=filters)

    elif analysis_type == "correlation":
        metric_b = params.get("metric_b")
        if not metric_b:
            return {"error": "Need two metrics for correlation"}
        return correlation_analysis(df, metric, metric_b, filters=filters, group_by=group_by)

    elif analysis_type == "summary":
        return summary_stats(df, metric, filters=filters, group_by=group_by)

    elif analysis_type == "normality":
        return normality_test(df, metric, group_col=group_by or category, filters=filters)

    elif analysis_type == "lookup":
        if not category:
            # Auto-pick: first categorical column that isn't the metric
            for c in df.columns:
                if c != metric and df[c].dtype == "object":
                    category = c
                    break
        if not category:
            return {"error": "No category column found for lookup"}
        return entity_lookup(df, metric, category, mode=params.get("lookup_mode", "max"), filters=filters)

    elif analysis_type == "entity_counts":
        return entity_counts(df, group_col=category)

    elif analysis_type == "pvm":
        # Auto-detect column names if not extracted
        def _find_col(candidates, cols):
            for cand in candidates:
                for col in cols:
                    if cand.lower() == col.lower():
                        return col
            return None

        rev_col = params.get("revenue_col") or _find_col(["rev", "revenue", "sales", "amount"], df.columns)
        q_col = params.get("qty_col") or _find_col(["qty", "quantity", "units", "volume"], df.columns)
        prod_col = params.get("product_col") or _find_col(["product_sku", "sku", "product", "item"], df.columns)
        cust_col = params.get("customer_col") or _find_col(["customer_id", "customer", "cust_id", "account"], df.columns)
        cost_c = params.get("cost_col") or _find_col(["cost", "cogs"], df.columns)
        margin_c = params.get("margin_col") or _find_col(["margin", "profit", "gross_margin"], df.columns)

        if not rev_col or not q_col or not prod_col:
            return {"error": f"PVM requires revenue, qty, and product columns. Found: rev={rev_col}, qty={q_col}, product={prod_col}"}

        return price_volume_mix(
            df,
            revenue_col=rev_col,
            qty_col=q_col,
            product_col=prod_col,
            customer_col=cust_col,
            period_a=params.get("period_a"),
            period_b=params.get("period_b"),
            cost_col=cost_c,
            margin_col=margin_c,
            filters=filters,
        )

    elif analysis_type == "period_comparison":
        return period_comparison(
            df, metric,
            period_a=params.get("period_a"),
            period_b=params.get("period_b"),
            stratify_by=params.get("stratify_by") or category,
        )

    else:
        return summary_stats(df, metric, filters=filters, group_by=group_by)


# ── Pass 2: Depth Analyses ──────────────────────────────────────────────────


def _run_depth(params: dict, _primary: dict, df, grain: dict, verbose: bool = True) -> list[dict]:
    """Pick and run 2 depth analyses based on Pass 1 results."""
    depth_results = []
    analysis_type = params["analysis_type"]
    metric = params["metric"]
    category = params.get("category")
    columns = list(df.columns)
    entity_col = grain.get("entity_col")
    time_col = grain.get("time_col")

    # Exclude ID, time, and already-used columns from candidate dimensions
    exclude = {entity_col, time_col, metric, category}
    cat_candidates = [
        c for c in columns
        if c not in exclude
        and not any(k in c.upper() for k in ("COST", "AMOUNT", "TOTAL", "INCOME", "AVG", "_ID", "KEY"))
        and df[c].nunique() > 1
        and df[c].nunique() <= 100
    ]

    if verbose:
        print(f"\nPass 2 — Depth analyses")

    if analysis_type == "distribution":
        # Depth 1: Statistical test on the same metric + category
        if category:
            if verbose:
                print(f"  Depth 1: comparison({metric}, {category})")
            depth_results.append(group_comparison(df, metric, category))

        # Depth 2: Distribution by a second dimension
        if cat_candidates:
            second_dim = cat_candidates[0]
            if verbose:
                print(f"  Depth 2: distribution({metric}, {second_dim})")
            depth_results.append(distribution_by_category(df, metric, second_dim))

    elif analysis_type == "comparison":
        # Depth 1: Full distribution view
        if category:
            if verbose:
                print(f"  Depth 1: distribution({metric}, {category})")
            depth_results.append(distribution_by_category(df, metric, category))

        # Depth 2: Entity counts by the group
        if category:
            if verbose:
                print(f"  Depth 2: entity_counts({category})")
            depth_results.append(entity_counts(df, group_col=category))

    elif analysis_type == "trend":
        # Depth 1: Summary stats for the metric
        if verbose:
            print(f"  Depth 1: summary({metric})")
        depth_results.append(summary_stats(df, metric))

        # Depth 2: Distribution by a dimension
        if cat_candidates:
            dim = cat_candidates[0]
            if verbose:
                print(f"  Depth 2: distribution({metric}, {dim})")
            depth_results.append(distribution_by_category(df, metric, dim))

    elif analysis_type == "correlation":
        # Depth 1 & 2: Summary of each metric
        if verbose:
            print(f"  Depth 1: summary({metric})")
        depth_results.append(summary_stats(df, metric))
        metric_b = params.get("metric_b")
        if metric_b:
            if verbose:
                print(f"  Depth 2: summary({metric_b})")
            depth_results.append(summary_stats(df, metric_b))

    elif analysis_type == "period_comparison":
        # Depth 1: Overall trend for context
        if verbose:
            print(f"  Depth 1: trend({metric})")
        depth_results.append(trend_over_time(df, metric))

        # Depth 2: Summary stats for the metric
        if verbose:
            print(f"  Depth 2: summary({metric})")
        depth_results.append(summary_stats(df, metric))

    elif analysis_type == "entity_counts":
        # Depth 1: Summary stats
        cost_cols = [c for c in columns if "COST" in c.upper()]
        if cost_cols:
            if verbose:
                print(f"  Depth 1: summary({cost_cols[0]})")
            depth_results.append(summary_stats(df, cost_cols[0]))

        # Depth 2: Distribution by a dimension
        if category and cost_cols:
            if verbose:
                print(f"  Depth 2: distribution({cost_cols[0]}, {category})")
            depth_results.append(distribution_by_category(df, cost_cols[0], category))

    else:
        # Fallback: summary + entity counts
        if verbose:
            print(f"  Depth 1: summary({metric})")
        depth_results.append(summary_stats(df, metric))
        if verbose:
            print(f"  Depth 2: entity_counts()")
        depth_results.append(entity_counts(df))

    return depth_results


# ── Narrate: LLM writes English from JSON ───────────────────────────────────


NARRATE_SYSTEM = """You are a senior healthcare data analyst writing a concise analytics report.

You will receive pre-computed analysis results as JSON. Your ONLY job is to narrate these numbers clearly.

RULES:
1. Every number you write MUST come from the JSON results. Copy-paste — do not estimate or round differently.
2. The JSON includes a "comparison.direction" field. USE IT VERBATIM for any higher/lower statements.
3. Do NOT invent additional statistics. If a number isn't in the JSON, don't state it.
4. Structure the report with: Executive Summary, Key Finding, Deeper Insights, Data Notes, Next Steps.
5. Keep it under 800 words. Lead with insight, not methodology.
6. State the grain and aggregation approach.
"""


async def _narrate(
    question: str,
    grain_context: str,
    primary: dict,
    depth: list[dict],
    verbose: bool = True,
) -> tuple[str, float | None]:
    """Single LLM call to narrate all results into a report.

    Returns (report_text, cost_usd).
    """
    # Build the narration prompt with all results
    results_json = json.dumps({
        "primary_analysis": primary,
        "depth_analyses": depth,
    }, indent=2, default=str)

    user_prompt = f"""Write an analytics report for this question:

**Question:** {question}

**Data Profile:**
{grain_context}

**Pre-computed Results (every number in your report MUST come from here):**

```json
{results_json}
```

Write the report now. Remember: copy numbers from the JSON, use comparison.direction verbatim, do not invent statistics."""

    response = _anthropic_client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=NARRATE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = _collect_text(response)
    if verbose:
        print(text)

    return text, None


# ── ML Workflow — Autoresearch-Inspired Experiment Loop ───────────────────────

ML_MAX_ROUNDS = 6
ML_SCORE_THRESHOLD = 0.65
ML_PLATEAU_TOLERANCE = 0.002  # stop if best hasn't improved by this much in 2 rounds


async def _run_ml_question(
    question: str,
    data_path: str,
    skill_name: str,
    grain_context: str,
    enable_loop: bool,
    verbose: bool,
) -> tuple[str, float | None, int, int]:
    """Run ML workflow using autoresearch-style experiment loop.

    Pattern (from karpathy/autoresearch):
      Round 1: Agent profiles data + generates initial pipeline code
      Round 2+: Agent receives best code + experiment log, makes ONE targeted change
      Each round: orchestrator runs pipeline, measures, keeps or discards
      Loop until: score >= threshold, max rounds, or plateau detected
    """
    from tools.ml_tools import run_pipeline_handler
    import pandas as pd

    t0 = time.time()
    total_cost = 0.0
    total_turns = 0

    # Compute target stats for the agent (avoids needing profile_data call)
    target_col = _extract_target(question, data_path)
    try:
        _df = pd.read_csv(data_path, usecols=[target_col])
        target_balance = float(_df[target_col].mean())
        target_counts = _df[target_col].value_counts().to_dict()
        target_info = f"\nTarget: {target_col} | Balance: {target_balance:.2%} positive | Counts: {target_counts}"
        del _df
    except Exception:
        target_info = ""

    # ── Experiment state (like autoresearch's results.tsv) ────────────
    experiment_log: list[dict] = []
    best_code: str | None = None
    best_score: float = 0.0
    best_metrics: dict = {}
    best_shap: list = []

    max_rounds = ML_MAX_ROUNDS if enable_loop else 1
    plateau_count = 0

    for round_num in range(1, max_rounds + 1):
        if verbose:
            print(f"\n{'=' * 60}")
            print(f"  EXPERIMENT ROUND {round_num}/{max_rounds}")
            print(f"  Best score so far: {best_score:.4f}")
            print(f"{'=' * 60}")

        # ── Build prompt for this round ───────────────────────────────
        if round_num == 1:
            prompt = _build_ml_initial_prompt(question, data_path, grain_context + target_info)
        else:
            prompt = _build_ml_improve_prompt(
                question, data_path, grain_context,
                best_code, best_score, best_metrics, experiment_log,
            )

        # ── Run agent (short call — profile/generate or modify code) ──
        code, _, cost, turns = await _run_ml_agent_round(
            prompt, data_path, grain_context, verbose,
        )
        total_cost += cost or 0.0
        total_turns += turns

        if not code:
            if verbose:
                print(f"  Round {round_num}: Agent did not return pipeline code. Skipping.")
            experiment_log.append({
                "round": round_num,
                "score": 0.0,
                "status": "crash",
                "description": "No pipeline code generated",
            })
            continue

        # ── Run pipeline (like autoresearch's `uv run train.py`) ──────
        if verbose:
            print(f"\n  Running pipeline...")

        result = await run_pipeline_handler({
            "code": code,
            "data_path": data_path,
            "target_col": _extract_target(question, data_path),
        })

        # Parse result
        result_text = result["content"][0]["text"]
        result_data = json.loads(result_text)

        if not result_data.get("success"):
            error = result_data.get("error", "Unknown error")
            stderr = result_data.get("stderr", "")
            if verbose:
                print(f"  CRASH: {error}")
                if stderr:
                    print(f"  stderr: {stderr[:300]}")
            experiment_log.append({
                "round": round_num,
                "score": 0.0,
                "status": "crash",
                "description": f"Pipeline failed: {error[:100]}",
            })
            continue

        # ── Measure (like autoresearch's `grep val_bpb run.log`) ──────
        metrics = result_data.get("metrics", {})
        score = result_data.get("composite_score", 0.0)
        shap = result_data.get("shap_features", [])

        if verbose:
            print(f"  Score: {score:.4f}  (best: {best_score:.4f})")
            print(f"  {result_data.get('score_breakdown', '')}")

        # ── Keep or discard (the core autoresearch pattern) ───────────
        if score > best_score:
            improvement = score - best_score
            best_score = score
            best_code = code
            best_metrics = metrics
            best_shap = shap
            plateau_count = 0
            if verbose:
                print(f"  KEEP — improved by {improvement:.4f}")
            experiment_log.append({
                "round": round_num,
                "score": score,
                "status": "keep",
                "description": _summarize_code_change(code, round_num),
                "metrics": {k: round(v, 4) if isinstance(v, float) else v
                           for k, v in metrics.items()},
            })
        else:
            plateau_count += 1
            if verbose:
                print(f"  DISCARD — score {score:.4f} <= best {best_score:.4f}")
            experiment_log.append({
                "round": round_num,
                "score": score,
                "status": "discard",
                "description": _summarize_code_change(code, round_num),
            })

        # ── Check stopping conditions ─────────────────────────────────
        if best_score >= ML_SCORE_THRESHOLD:
            if verbose:
                print(f"\n  Score {best_score:.4f} >= threshold {ML_SCORE_THRESHOLD}. Stopping.")
            break

        if plateau_count >= 2 and round_num >= 3:
            if verbose:
                print(f"\n  Plateau detected (no improvement in {plateau_count} rounds). Stopping.")
            break

    # ── Final report (narrate results like the analytics 2-pass) ──────
    if verbose:
        print(f"\n{'=' * 60}")
        print("  GENERATING FINAL REPORT")
        print(f"{'=' * 60}")

    report = await _narrate_ml_results(
        question, grain_context, best_metrics, best_shap, experiment_log, verbose,
    )

    # Save final pipeline code + experiment log alongside the report
    _save_ml_artifacts(best_code, experiment_log, best_metrics)

    duration_ms = int((time.time() - t0) * 1000)
    return report, total_cost, total_turns, duration_ms


async def _run_ml_agent_round(
    prompt: str,
    data_path: str,
    grain_context: str,
    verbose: bool,
) -> tuple[str | None, str, float | None, int]:
    """Run one agent round. Returns (pipeline_code, text, cost, turns).

    The agent profiles data and generates pipeline code as text.
    The orchestrator (not the agent) runs the pipeline — this avoids
    double execution and gives us control over keep/discard.
    """
    import re

    # No MCP tools needed — data profile is in the system prompt,
    # and the orchestrator runs the pipeline. This makes each round faster.
    ml_system = (
        f"You are an ML engineer. Data file: {data_path}\n\n"
        f"## Data Profile\n{grain_context}\n\n"
        "IMPORTANT: Output your pipeline code in a ```python code block. "
        "Do NOT call any tools — just generate the code. "
        "The orchestrator runs the pipeline for you. "
        "Use DATA_PATH = 'PLACEHOLDER' and TARGET_COL = 'PLACEHOLDER'."
    )

    response = _anthropic_client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=ml_system,
        messages=[{"role": "user", "content": prompt}],
    )
    full_text = _collect_text(response)
    cost_usd: float | None = None
    num_turns = 1
    if verbose:
        print(full_text)

    # Extract pipeline code from ```python ... ``` blocks in agent output
    code_blocks = re.findall(r"```python\s*\n(.*?)```", full_text, re.DOTALL)

    # Find the largest code block that looks like a pipeline
    extracted_code = None
    for block in sorted(code_blocks, key=len, reverse=True):
        # Accept any substantial code block that imports ML libraries
        if len(block) > 200 and any(k in block for k in ("train_test_split", "LGBMClassifier", "lgb.", "xgb.", "RandomForest")):
            extracted_code = block.strip()
            # Ensure it writes metrics.json (required by run_pipeline_handler)
            if "metrics.json" not in extracted_code:
                extracted_code = _inject_metrics_output(extracted_code)
            break

    return extracted_code, full_text, cost_usd, num_turns


def _inject_metrics_output(code: str) -> str:
    """Inject metrics.json + shap_features.json output into pipeline code.

    The run_pipeline_handler requires metrics.json to exist after execution.
    If the agent forgot to write it, we append the output code.
    """
    import textwrap

    # Add json import if missing
    if "import json" not in code:
        code = "import json\n" + code

    # Add pathlib import if missing
    if "from pathlib import Path" not in code:
        code = "from pathlib import Path\n" + code

    # Append metrics output — uses sklearn imports that should already be in the code
    metrics_code = textwrap.dedent("""

    # ── Write metrics.json (required by orchestrator) ──
    from sklearn.metrics import roc_auc_score as _auc, f1_score as _f1, accuracy_score as _acc, precision_score as _prec, recall_score as _rec
    import numpy as _np
    _metrics_out = {
        "auc": float(_auc(y_test, y_pred_proba)),
        "f1": float(_f1(y_test, y_pred, average='weighted', zero_division=0)),
        "accuracy": float(_acc(y_test, y_pred)),
        "precision": float(_prec(y_test, y_pred, average='weighted', zero_division=0)),
        "recall": float(_rec(y_test, y_pred, average='weighted', zero_division=0)),
    }
    Path("metrics.json").write_text(json.dumps(_metrics_out, indent=2))
    print("Wrote metrics.json")

    # ── Write shap_features.json if possible ──
    try:
        import shap
        _explainer = shap.TreeExplainer(model)
        _sv = _explainer.shap_values(X_test.iloc[:min(500, len(X_test))])
        if isinstance(_sv, list):
            _arr = _np.abs(_sv[1]) if len(_sv) >= 2 else _np.abs(_sv[0])
        elif _sv.ndim == 3:
            _arr = _np.abs(_sv[:, :, 1])
        else:
            _arr = _np.abs(_sv)
        _mean_imp = _arr.mean(axis=0)
        _top30_idx = _mean_imp.argsort()[-30:][::-1]
        _features = X_test.columns if hasattr(X_test, 'columns') else [f"f{i}" for i in range(len(_mean_imp))]
        _shap_features = [{"feature": str(_features[i]), "mean_abs_shap": float(_mean_imp[i])} for i in _top30_idx]
        Path("shap_features.json").write_text(json.dumps(_shap_features, indent=2))
        _metrics_out["explainability_coverage"] = float(sorted(_mean_imp)[-10:].sum() / _mean_imp.sum()) if _mean_imp.sum() > 0 else 0.0
        Path("metrics.json").write_text(json.dumps(_metrics_out, indent=2))
        print("Wrote shap_features.json")
    except Exception as _e:
        print(f"SHAP failed (non-fatal): {_e}")
    """)

    return code + metrics_code


def _build_ml_initial_prompt(question: str, data_path: str, grain_context: str) -> str:
    """Round 1 prompt: profile data and generate initial pipeline."""
    return f"""The user asked: "{question}"
Data file: {data_path}

This is Round 1 of an iterative ML experiment loop. The data profile (including target balance) is provided above — do NOT call profile_data.

Your job: Generate a complete classification pipeline and output it in a ```python code block.
Do NOT call run_pipeline — the orchestrator runs it.

Use DATA_PATH = 'PLACEHOLDER' and TARGET_COL = 'PLACEHOLDER' (they get injected automatically).
The pipeline MUST write metrics.json with keys: auc, f1, accuracy, precision, recall.
Optionally write shap_features.json with top-30 features.

This is the BASELINE run. Focus on getting a clean, working pipeline — not on optimization yet.
Use LightGBM with sensible defaults. Keep it simple."""


def _build_ml_improve_prompt(
    question: str,
    data_path: str,
    grain_context: str,
    best_code: str,
    best_score: float,
    best_metrics: dict,
    experiment_log: list[dict],
) -> str:
    """Round 2+ prompt: make ONE targeted improvement to the best code."""
    log_text = "\n".join(
        f"  Round {e['round']}: score={e['score']:.4f} [{e['status']}] — {e['description']}"
        for e in experiment_log
    )
    metrics_text = json.dumps(
        {k: round(v, 4) if isinstance(v, float) else v for k, v in best_metrics.items()},
        indent=2,
    )

    return f"""The user asked: "{question}"
Data file: {data_path}

This is an IMPROVEMENT round in an iterative ML experiment loop.

## Experiment History
{log_text}

## Current Best Score: {best_score:.4f}
## Current Best Metrics:
{metrics_text}

## Current Best Code:
```python
{best_code}
```

## Your Task
Make ONE targeted improvement to the code above and output the FULL modified pipeline
in a ```python code block. Do NOT call run_pipeline — the orchestrator runs it.
Use DATA_PATH = 'PLACEHOLDER' and TARGET_COL = 'PLACEHOLDER'.
The pipeline MUST write metrics.json with keys: auc, f1, accuracy, precision, recall.

**Rules (from autoresearch pattern):**
- Make ONE change at a time. Do NOT rewrite the entire pipeline.
- Look at the experiment history — do NOT repeat something that was already tried and discarded.
- Focus on the weakest metric component for the biggest potential gain.

**Ideas to try (pick ONE):**
- Switch model type (lgbm → xgb → rf)
- Tune hyperparameters (learning_rate, n_estimators, num_leaves, max_depth)
- Add feature selection (shap_top_k, correlation filtering, mutual_info)
- Handle class imbalance differently (scale_pos_weight, SMOTE)
- Add preprocessing (log transforms on skewed features)
- Remove noisy features (high-cardinality, low-variance)"""


def _extract_target(question: str, data_path: str) -> str:
    """Extract target column from question or default."""
    q_lower = question.lower()
    # Look for "predicting X" or "predict X" pattern
    for pattern in ["predicting ", "predict "]:
        if pattern in q_lower:
            after = q_lower.split(pattern, 1)[1]
            # Take the first word/phrase that looks like a column name
            candidate = after.split()[0].strip(".,;:!?\"'").upper()
            if candidate:
                return candidate
    return "TARGET_HIGH_COST_FLAG"


def _summarize_code_change(code: str, round_num: int) -> str:
    """Extract a short description of the pipeline config from code."""
    desc_parts = []
    if "MODEL_TYPE" in code:
        for line in code.split("\n"):
            if line.strip().startswith("MODEL_TYPE"):
                val = line.split("=", 1)[1].strip().strip('"\'# ')
                desc_parts.append(f"model={val.split()[0]}")
                break
    if "FEATURE_SELECTION" in code:
        for line in code.split("\n"):
            if line.strip().startswith("FEATURE_SELECTION") and "FEATURE_SELECTION_K" not in line:
                val = line.split("=", 1)[1].strip().strip('"\'# ')
                if val != "none":
                    desc_parts.append(f"feat_sel={val.split()[0]}")
                break
    if round_num == 1:
        desc_parts.insert(0, "baseline")
    return " | ".join(desc_parts) if desc_parts else f"round {round_num}"


def _save_ml_artifacts(
    best_code: str | None,
    experiment_log: list[dict],
    best_metrics: dict,
) -> None:
    """Save final pipeline code and experiment log to output/."""
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save best pipeline code
    if best_code:
        code_path = OUTPUT_DIR / f"{ts}_best_pipeline.py"
        code_path.write_text(best_code)
        print(f"Pipeline code saved: {code_path}")

    # Save experiment log (like autoresearch's results.tsv)
    log_path = OUTPUT_DIR / f"{ts}_experiment_log.json"
    log_data = {
        "experiment_log": experiment_log,
        "best_metrics": best_metrics,
        "total_rounds": len(experiment_log),
        "kept_rounds": sum(1 for e in experiment_log if e["status"] == "keep"),
    }
    log_path.write_text(json.dumps(log_data, indent=2, default=str))
    print(f"Experiment log saved: {log_path}")


async def _narrate_ml_results(
    question: str,
    grain_context: str,
    best_metrics: dict,
    best_shap: list,
    experiment_log: list[dict],
    verbose: bool,
) -> str:
    """Single LLM call to narrate ML results into a report."""
    log_text = "\n".join(
        f"  Round {e['round']}: score={e['score']:.4f} [{e['status']}] — {e['description']}"
        for e in experiment_log
    )

    results_json = json.dumps({
        "best_metrics": best_metrics,
        "top_shap_features": best_shap[:15],
        "experiment_log": experiment_log,
        "total_rounds": len(experiment_log),
        "kept_rounds": sum(1 for e in experiment_log if e["status"] == "keep"),
    }, indent=2, default=str)

    user_prompt = f"""Write an ML model report for this question:

**Question:** {question}

**Data Profile:**
{grain_context}

**Experiment Log:**
{log_text}

**Best Model Results:**
```json
{results_json}
```

Write the report with these sections:
1. **Executive Summary** — What was built, final score, key finding
2. **Experiment Journey** — What was tried, what worked, what didn't (use the experiment log)
3. **Model Performance** — Metrics table (AUC, F1, accuracy, precision, recall, composite)
4. **Top Drivers (SHAP)** — Top 10 features with business interpretation
5. **Model Assessment** — Production readiness, limitations, next steps

Copy all numbers from the JSON. Do not invent statistics.
Write the FULL report now — do not say you will write it, just write it directly."""

    ML_NARRATE_SYSTEM = """You are a senior data scientist writing an ML model report.
You receive pre-computed metrics, SHAP features, and an experiment log.
Your job is to narrate these results clearly and provide business interpretation.
Every number must come from the provided JSON. Do not estimate or invent statistics.

IMPORTANT: Write the full report immediately in your response. Do NOT say "let me write" or "I'll create" — just write the report directly. Do NOT use any tools."""

    response = _anthropic_client.messages.create(
        model=_MODEL,
        max_tokens=4096,
        system=ML_NARRATE_SYSTEM,
        messages=[{"role": "user", "content": user_prompt}],
    )
    text = _collect_text(response)
    if verbose:
        print(text)
    return text


# ── Core Runner ──────────────────────────────────────────────────────────────


async def run_question(
    question: str,
    data_path: str,
    enable_loop: bool = True,
    verbose: bool = True,
) -> str:
    """Run a question through the 2-pass analytics engine.

    Pass 1: Deterministic primary analysis
    Pass 2: Deterministic depth analyses (2 additional)
    Narrate: Single LLM call to write the report
    """
    t0 = time.time()

    # 1. Route question
    skills = discover_analytics_skills()
    skill_name = route_question(question, skills)

    if verbose:
        print(f"Question: {question}")
        print(f"Data: {data_path}")
        print(f"Routed to: {skill_name or '(direct)'}")
        print("-" * 60)

    # 2. Profile data
    grain_context = build_grain_context(data_path)
    if verbose:
        print(f"\n{grain_context}\n")
        print("-" * 60)

    # 3. ML workflow uses the old agent-based approach
    is_ml = skill_name in ("classification",)
    if is_ml:
        report, cost_usd, num_turns, duration_ms = await _run_ml_question(
            question, data_path, skill_name, grain_context, enable_loop, verbose
        )
        _save_report(question, data_path, skill_name, report, cost_usd, num_turns, duration_ms)
        return report

    # 4. Load data and detect grain
    df = _load_df(data_path)
    grain = _detect_grain(df)
    columns = list(df.columns)

    # Enrich grain with categorical values for value-based column matching
    # (e.g., "male"/"female" → SEX, "active"/"inactive" → CURRENTLY_ACTIVE)
    cat_values = {}
    for col in columns:
        if col == grain.get("entity_col") or col == grain.get("time_col"):
            continue
        nunique = df[col].nunique()
        if 2 <= nunique <= 20:
            cat_values[col] = df[col].dropna().unique().tolist()
    grain["_cat_values"] = cat_values

    # 5. Extract parameters
    params = await _extract_params(question, skill_name, columns, grain)
    if verbose:
        print(f"\nExtracted: {json.dumps(params, indent=2)}")

    # 6. Pass 1 — Primary analysis (deterministic)
    primary = _run_primary(params, df, verbose)
    if verbose:
        # Print key headline from primary result
        comp = primary.get("comparison")
        if comp:
            print(f"\n  → {comp.get('direction', '')}")
        interp = primary.get("interpretation")
        if interp:
            print(f"\n  → {interp.get('summary', '')}")

    # 7. Pass 2 — Depth analyses (deterministic)
    if enable_loop:
        depth = _run_depth(params, primary, df, grain, verbose)
    else:
        depth = []

    # 8. Narrate (single LLM call)
    if verbose:
        print(f"\n{'=' * 60}")
        print("Narrating report...")
        print(f"{'=' * 60}\n")

    report, narrate_cost = await _narrate(question, grain_context, primary, depth, verbose)

    duration_ms = int((time.time() - t0) * 1000)

    # 9. Save
    _save_report(
        question=question,
        data_path=data_path,
        skill_name=skill_name,
        report=report,
        cost_usd=narrate_cost,
        num_turns=1,
        duration_ms=duration_ms,
    )

    return report


# ── Report Saving ────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(__file__).resolve().parent / "output"


def _save_report(
    question: str,
    data_path: str,
    skill_name: str | None,
    report: str,
    cost_usd: float | None = None,
    num_turns: int = 0,
    duration_ms: int = 0,
) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    skill = skill_name or "direct"
    filepath = OUTPUT_DIR / f"{ts}_{skill}_report.md"

    header = [
        "# Analytics Report", "",
        f"**Question:** {question}",
        f"**Data:** {data_path}",
        f"**Skill:** {skill}",
        f"**Timestamp:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
    ]
    if cost_usd is not None:
        header.append(f"**Cost:** ${cost_usd:.4f}")
    header.append(f"**Turns:** {num_turns} | **Duration:** {duration_ms}ms")
    header += ["", "---", ""]

    filepath.write_text("\n".join(header) + report)
    print(f"\nReport saved: {filepath}")
    return filepath


# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Analytics Orchestrator — ask questions about your data"
    )
    parser.add_argument("question", help="Natural language question about the data")
    parser.add_argument("--data", required=True, help="Path to CSV file")
    parser.add_argument(
        "--quick", action="store_true",
        help="Pass 1 only — skip depth analyses and ML experiment loop",
    )
    parser.add_argument(
        "--quiet", action="store_true", help="Suppress verbose output",
    )
    args = parser.parse_args()

    asyncio.run(
        run_question(
            args.question,
            data_path=args.data,
            enable_loop=not args.quick,
            verbose=not args.quiet,
        )
    )


if __name__ == "__main__":
    main()
