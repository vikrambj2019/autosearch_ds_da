"""
MCP server wiring — connects tool handlers to Claude SDK MCP protocol.

Two servers:
  - data: profiling, code execution, column validation, deterministic analytics
  - ml:   pipeline execution, composite scoring
"""

from claude_agent_sdk import create_sdk_mcp_server, tool

from .data_tools import (
    profile_data_handler,
    run_code_handler,
    validate_cols_handler,
    distribution_handler,
    trend_handler,
    comparison_handler,
    correlation_handler,
    summary_handler,
    entity_counts_handler,
    period_comparison_handler,
)
from .ml_tools import (
    run_pipeline_handler,
    score_metrics_handler,
)

# ── Data tools ────────────────────────────────────────────────────────────────

_profile_data = tool(
    name="profile_data",
    description=(
        "Profile a dataset: schema, grain detection (entity key + time dimension), "
        "column statistics, sample rows, and data quality warnings. "
        "ALWAYS call this first before any analysis."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "sample_rows": {
                "type": "integer",
                "description": "Number of sample rows to return (default: 5)",
            },
        },
        "required": ["data_path"],
    },
)(profile_data_handler)


_run_code = tool(
    name="run_code",
    description=(
        "Execute pandas analysis code in a sandboxed environment. "
        "The CSV is pre-loaded as `df`. Available: pd, np, stats (scipy.stats). "
        "Do NOT write import statements or pd.read_csv(). "
        "PREFER the dedicated analytics tools (distribution, trend, comparison, "
        "correlation, summary, entity_counts) over writing free-form code. "
        "Only use run_code for questions the dedicated tools cannot answer."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Pandas/numpy code to execute. df is pre-loaded.",
            },
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "return_format": {
                "type": "string",
                "description": "'table' (default), 'scalar', 'json', or 'markdown'",
            },
        },
        "required": ["code", "data_path"],
    },
)(run_code_handler)


_validate_cols = tool(
    name="validate_cols",
    description=(
        "Fuzzy-match user column references to actual column names in a dataset. "
        "Use this to resolve ambiguous column names from user queries "
        "(e.g., 'county' -> 'COUNTY_LATEST', 'ER cost' -> 'MONTHLY_ER_COST')."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "user_columns": {
                "type": "string",
                "description": "Comma-separated column names from the user query",
            },
        },
        "required": ["data_path", "user_columns"],
    },
)(validate_cols_handler)


# ── Deterministic analytics tools ─────────────────────────────────────────────

_distribution = tool(
    name="distribution",
    description=(
        "Compute distribution of a numeric metric grouped by a categorical variable. "
        "Deterministic and grain-aware: automatically collapses panel data to entity-level "
        "before computing stats. Handles high-cardinality categories (top N + Other). "
        "Returns per-group statistics (mean, median, std, percentiles), totals, cost shares, "
        "and a pre-computed comparison showing which group is higher/lower. "
        "Use this for questions like 'distribution of cost by active status' or "
        "'breakdown of ER cost by county'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "metric": {
                "type": "string",
                "description": "Numeric column to analyze (e.g., 'MONTHLY_TOTAL_COST')",
            },
            "category": {
                "type": "string",
                "description": "Categorical column to group by (e.g., 'CURRENTLY_ACTIVE')",
            },
            "top_n": {
                "type": "integer",
                "description": "Max categories to show (default: 10, rest collapsed to 'Other')",
            },
        },
        "required": ["data_path", "metric", "category"],
    },
)(distribution_handler)


_trend = tool(
    name="trend",
    description=(
        "Compute trend of a numeric metric over time periods. "
        "Returns per-period mean/total, month-over-month change, overall direction, "
        "and linear regression stats (slope, R², p-value). "
        "Optionally stratify by a category (e.g., trend by county). "
        "Use this for questions like 'cost trend over time' or 'monthly trend by sex'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "metric": {
                "type": "string",
                "description": "Numeric column to trend (e.g., 'MONTHLY_TOTAL_COST')",
            },
            "stratify_by": {
                "type": "string",
                "description": "Optional: categorical column to stratify the trend (e.g., 'SEX')",
            },
            "top_n": {
                "type": "integer",
                "description": "Max categories when stratifying (default: 10)",
            },
        },
        "required": ["data_path", "metric"],
    },
)(trend_handler)


_comparison = tool(
    name="comparison",
    description=(
        "Statistical comparison of a numeric metric between two groups. "
        "Grain-aware: collapses panel data to entity-level before testing. "
        "Runs t-test, Mann-Whitney U, and computes Cohen's d effect size. "
        "Returns pre-computed direction (which group is higher). "
        "Use this for questions like 'is cost different between male and female?' "
        "or 'compare ER cost between active and inactive members'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "metric": {
                "type": "string",
                "description": "Numeric column to compare (e.g., 'MONTHLY_TOTAL_COST')",
            },
            "group_col": {
                "type": "string",
                "description": "Categorical column defining groups (e.g., 'SEX')",
            },
            "group_a": {
                "type": "string",
                "description": "Optional: value for group A (auto-detected if omitted)",
            },
            "group_b": {
                "type": "string",
                "description": "Optional: value for group B (auto-detected if omitted)",
            },
        },
        "required": ["data_path", "metric", "group_col"],
    },
)(comparison_handler)


_correlation = tool(
    name="correlation",
    description=(
        "Correlation between two numeric columns. "
        "Grain-aware: collapses panel data to entity-level means first. "
        "Returns Pearson r, Spearman rho, p-values, and interpretation "
        "(strength: strong/moderate/weak/negligible, direction: positive/negative). "
        "Use this for questions like 'does income correlate with cost?'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "metric_a": {
                "type": "string",
                "description": "First numeric column (e.g., 'MEDIAN_HOUSEHOLD_INCOME_LATEST')",
            },
            "metric_b": {
                "type": "string",
                "description": "Second numeric column (e.g., 'MONTHLY_TOTAL_COST')",
            },
        },
        "required": ["data_path", "metric_a", "metric_b"],
    },
)(correlation_handler)


_summary = tool(
    name="summary",
    description=(
        "Summary statistics for a single numeric metric. "
        "Grain-aware: collapses panel data to entity-level means. "
        "Returns mean, median, std, percentiles (p25/p50/p75/p90/p95/p99), "
        "skewness, zero percentage, and total sum. "
        "Use this for questions like 'what are the summary stats for cost?' "
        "or 'what is the 90th percentile of ER cost?'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "metric": {
                "type": "string",
                "description": "Numeric column to summarize (e.g., 'MONTHLY_TOTAL_COST')",
            },
        },
        "required": ["data_path", "metric"],
    },
)(summary_handler)


_entity_counts = tool(
    name="entity_counts",
    description=(
        "Count unique entities, optionally grouped by a category. "
        "Grain-aware: uses nunique(entity_col) for panel data, len(df) for cross-sectional. "
        "Handles high cardinality (top N + Other). "
        "Use this for questions like 'how many members?' or "
        "'how many members per county?'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "group_col": {
                "type": "string",
                "description": "Optional: categorical column to group counts by",
            },
            "top_n": {
                "type": "integer",
                "description": "Max categories to show (default: 10)",
            },
        },
        "required": ["data_path"],
    },
)(entity_counts_handler)


_period_comparison = tool(
    name="period_comparison",
    description=(
        "Entity-level comparison between two time periods in panel data. "
        "Shows: aggregate delta (mean/sum change), cohort matching (new/churned/retained members), "
        "member movement (% who increased/decreased/stayed flat), "
        "transition matrix (cost bucket shifts), top movers, and segment drivers. "
        "This is the analysis you can ONLY do with panel data — it answers "
        "'what changed between months and who drove the change?' "
        "Use for questions like 'compare July vs September costs', "
        "'what changed between months', 'which members drove the cost increase'."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "data_path": {
                "type": "string",
                "description": "Path to the CSV or parquet file",
            },
            "metric": {
                "type": "string",
                "description": "Numeric column to compare (e.g., 'MONTHLY_TOTAL_COST')",
            },
            "period_a": {
                "type": "string",
                "description": "First period (e.g., '2025-07-01' or 'July'). Default: second-to-last period.",
            },
            "period_b": {
                "type": "string",
                "description": "Second period (e.g., '2025-09-01' or 'September'). Default: last period.",
            },
            "stratify_by": {
                "type": "string",
                "description": "Optional: categorical column to decompose drivers (e.g., 'COUNTY_LATEST')",
            },
            "top_n": {
                "type": "integer",
                "description": "Max categories when stratifying (default: 10)",
            },
            "top_movers": {
                "type": "integer",
                "description": "Number of top movers to show (default: 10)",
            },
        },
        "required": ["data_path", "metric"],
    },
)(period_comparison_handler)


def create_data_server():
    """Create the in-process MCP server for data tools."""
    return create_sdk_mcp_server(
        name="data",
        version="1.0.0",
        tools=[
            _profile_data,
            _run_code,
            _validate_cols,
            _distribution,
            _trend,
            _comparison,
            _correlation,
            _summary,
            _entity_counts,
            _period_comparison,
        ],
    )


# ── ML tools ─────────────────────────────────────────────────────────────────

_run_pipeline = tool(
    name="run_pipeline",
    description=(
        "Execute a complete ML pipeline Python script. The script must write "
        "metrics.json (with auc, f1, accuracy, precision, recall, train_time, "
        "explainability_coverage) and optionally shap_features.json. "
        "Returns metrics + composite quality score. "
        "Use DATA_PATH = 'PLACEHOLDER' and TARGET_COL = 'PLACEHOLDER' in the code — "
        "they will be injected automatically."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "Complete Python ML pipeline code to execute",
            },
            "data_path": {
                "type": "string",
                "description": "Path to the training data CSV file",
            },
            "target_col": {
                "type": "string",
                "description": "Target column name (default: TARGET_HIGH_COST_FLAG)",
            },
            "timeout": {
                "type": "integer",
                "description": "Execution timeout in seconds (default: 300)",
            },
        },
        "required": ["code", "data_path"],
    },
)(run_pipeline_handler)


_score_metrics = tool(
    name="score_metrics",
    description=(
        "Compute a composite ML quality score from pipeline metrics. "
        "Score = 0.45*AUC + 0.20*F1 + 0.10*speed + 0.10*shap_coverage + 0.15*llm_explainability. "
        "Returns composite score, per-component breakdown, and scoring weights."
    ),
    input_schema={
        "type": "object",
        "properties": {
            "metrics": {
                "type": "object",
                "description": "Pipeline metrics dict (auc, f1, train_time, explainability_coverage, etc.)",
            },
            "weights": {
                "type": "object",
                "description": "Optional custom scoring weights",
            },
        },
        "required": ["metrics"],
    },
)(score_metrics_handler)


def create_ml_server():
    """Create the in-process MCP server for ML tools."""
    return create_sdk_mcp_server(
        name="ml",
        version="1.0.0",
        tools=[_run_pipeline, _score_metrics],
    )
