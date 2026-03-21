"""
MCP tools for ML pipeline execution and scoring.

run_pipeline: Execute a complete ML pipeline script and return metrics.
score_metrics: Compute composite score from pipeline metrics.
"""

from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

# Default weights — can be overridden per skill via manifest scoring config
DEFAULT_WEIGHTS = {
    "auc": 0.45,
    "f1": 0.20,
    "speed": 0.10,
    "shap_coverage": 0.10,
    "llm_explainability": 0.15,
}


def composite_score(metrics: dict, weights: dict | None = None) -> float:
    """Compute composite ML quality score from pipeline metrics.

    Components (all normalized to [0, 1]):
      - AUC (from metrics)
      - F1 weighted (from metrics)
      - Speed: 1 / (1 + log1p(train_time / 30))
      - SHAP coverage: top-10 SHAP importance share
      - LLM explainability: external LLM rating (default 0.5)
    """
    w = weights or DEFAULT_WEIGHTS

    auc = metrics.get("auc", 0.5)
    f1 = metrics.get("f1", 0.0)
    t = max(metrics.get("train_time", 999.0), 1.0)
    shap_cov = metrics.get("explainability_coverage", 0.0)
    llm_expl = metrics.get("llm_explainability_score", 0.5)

    speed = 1.0 / (1.0 + math.log1p(t / 30.0))

    return (
        w.get("auc", 0.45) * auc
        + w.get("f1", 0.20) * f1
        + w.get("speed", 0.10) * speed
        + w.get("shap_coverage", 0.10) * shap_cov
        + w.get("llm_explainability", 0.15) * llm_expl
    )


def score_breakdown(metrics: dict, weights: dict | None = None) -> str:
    """Human-readable breakdown of the composite score."""
    w = weights or DEFAULT_WEIGHTS
    auc = metrics.get("auc", 0.0)
    f1 = metrics.get("f1", 0.0)
    t = max(metrics.get("train_time", 999.0), 1.0)
    shap_cov = metrics.get("explainability_coverage", 0.0)
    llm_expl = metrics.get("llm_explainability_score", 0.5)
    speed = 1.0 / (1.0 + math.log1p(t / 30.0))
    total = composite_score(metrics, w)

    return (
        f"AUC={auc:.4f} F1={f1:.4f} speed={speed:.3f} "
        f"shap={shap_cov:.3f} llm_expl={llm_expl:.3f} "
        f"time={t:.1f}s → composite={total:.4f}"
    )


# ---------------------------------------------------------------------------
# Pipeline validation
# ---------------------------------------------------------------------------

# Required elements in a valid ML pipeline
_REQUIRED_ELEMENTS = ["metrics.json", "shap_features.json"]

# Blocked imports that shouldn't appear in ML pipeline code
_ML_BLOCKED = [
    "import socket",
    "import http",
    "import urllib",
    "import requests",
    "__import__",
]


def validate_pipeline_code(code: str) -> tuple[bool, str]:
    """Validate that ML pipeline code is syntactically correct and safe.

    Returns (is_valid, message).
    """
    # Syntax check
    try:
        ast.parse(code)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"

    # Must write metrics output
    if "metrics.json" not in code:
        return False, "Pipeline must write metrics.json"

    # Security check
    for blocked in _ML_BLOCKED:
        if blocked in code:
            return False, f"Blocked pattern: {blocked}"

    return True, "ok"


# ---------------------------------------------------------------------------
# MCP tool handlers
# ---------------------------------------------------------------------------

PIPELINE_TIMEOUT = 300  # 5 minutes


async def run_pipeline_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Execute an ML pipeline script and return its metrics.

    The pipeline must:
      - Accept DATA_PATH and TARGET_COL as defined constants
      - Write metrics.json to its working directory
      - Optionally write shap_features.json

    Input:
      - code: Complete Python pipeline code (string)
      - data_path: Path to training data CSV
      - target_col: Target column name
      - timeout: Execution timeout in seconds (default: 300)

    Output:
      - success: bool
      - metrics: dict (from metrics.json)
      - composite_score: float
      - score_breakdown: str
      - shap_features: list (from shap_features.json, if written)
      - execution_time_s: float
      - stdout: str (last 2000 chars)
      - stderr: str (last 2000 chars)
      - error: str (if failed)
    """
    code = args["code"]
    data_path = str(Path(args["data_path"]).resolve())  # Always use absolute path
    target_col = args.get("target_col", "TARGET_HIGH_COST_FLAG")
    timeout = args.get("timeout", PIPELINE_TIMEOUT)

    # Validate code
    is_valid, msg = validate_pipeline_code(code)
    if not is_valid:
        return {
            "content": [{"type": "text", "text": json.dumps({
                "success": False,
                "error": f"Pipeline validation failed: {msg}",
            }, indent=2)}],
            "is_error": True,
        }

    # Inject data path and target col into the code
    # Use string paths (not Path objects) to avoid missing import issues
    code = code.replace('DATA_PATH = "PLACEHOLDER"', f'DATA_PATH = "{data_path}"')
    code = code.replace("DATA_PATH = 'PLACEHOLDER'", f"DATA_PATH = '{data_path}'")
    code = code.replace('TARGET_COL = "PLACEHOLDER"', f'TARGET_COL = "{target_col}"')
    code = code.replace("TARGET_COL = 'PLACEHOLDER'", f"TARGET_COL = '{target_col}'")

    # Write to temp directory and execute
    with tempfile.TemporaryDirectory(prefix="ml_pipeline_") as tmpdir:
        tmpdir_path = Path(tmpdir)
        pipeline_file = tmpdir_path / "pipeline.py"
        pipeline_file.write_text(code)

        metrics_file = tmpdir_path / "metrics.json"
        shap_file = tmpdir_path / "shap_features.json"

        t0 = time.time()
        try:
            result = subprocess.run(
                [sys.executable, str(pipeline_file)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=str(tmpdir_path),
            )
            elapsed = time.time() - t0
        except subprocess.TimeoutExpired:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": False,
                    "error": f"Pipeline timed out after {timeout}s",
                    "execution_time_s": timeout,
                }, indent=2)}],
                "is_error": True,
            }

        stdout = result.stdout[-2000:] if result.stdout else ""
        stderr = result.stderr[-2000:] if result.stderr else ""

        if result.returncode != 0:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": False,
                    "error": f"Pipeline failed (exit code {result.returncode})",
                    "stderr": stderr,
                    "stdout": stdout,
                    "execution_time_s": round(elapsed, 1),
                }, indent=2)}],
                "is_error": True,
            }

        # Read metrics
        if not metrics_file.exists():
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": False,
                    "error": "Pipeline completed but metrics.json was not written",
                    "stdout": stdout,
                    "execution_time_s": round(elapsed, 1),
                }, indent=2)}],
                "is_error": True,
            }

        try:
            metrics = json.loads(metrics_file.read_text())
        except json.JSONDecodeError as e:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "success": False,
                    "error": f"metrics.json is invalid JSON: {e}",
                    "execution_time_s": round(elapsed, 1),
                }, indent=2)}],
                "is_error": True,
            }

        # Add train_time if not in metrics
        if "train_time" not in metrics:
            metrics["train_time"] = round(elapsed, 1)

        # Read SHAP features if present and compute explainability_coverage
        shap_features = None
        if shap_file.exists():
            try:
                shap_features = json.loads(shap_file.read_text())
            except json.JSONDecodeError:
                pass

        if shap_features and "explainability_coverage" not in metrics:
            # Normalize shap_features to a flat list of dicts
            # Agents write various formats: list of dicts, dict with nested list, list of strings
            features_list = shap_features
            if isinstance(shap_features, dict):
                # e.g. {"top_30_features": [...]} or {"features": [...]}
                for key in ("top_30_features", "features", "shap_features"):
                    if key in shap_features and isinstance(shap_features[key], list):
                        features_list = shap_features[key]
                        break
                else:
                    features_list = []

            if features_list and isinstance(features_list, list):
                if isinstance(features_list[0], dict):
                    importances = [f.get("importance", f.get("mean_abs_shap", 0)) for f in features_list]
                    total = sum(importances)
                    top10 = sum(sorted(importances, reverse=True)[:10])
                    metrics["explainability_coverage"] = round(top10 / total, 4) if total > 0 else 0.0
                else:
                    # Feature names only — count-based coverage
                    metrics["explainability_coverage"] = round(min(10, len(features_list)) / max(len(features_list), 1), 4)
                # Normalize shap_features for output
                shap_features = features_list

        # Score
        cs = composite_score(metrics)
        bd = score_breakdown(metrics)

        output = {
            "success": True,
            "metrics": metrics,
            "composite_score": round(cs, 4),
            "score_breakdown": bd,
            "execution_time_s": round(elapsed, 1),
            "stdout": stdout[-500:],
        }
        if shap_features:
            output["shap_features"] = shap_features[:20]  # Top 20 only

        return {
            "content": [{"type": "text", "text": json.dumps(output, indent=2, default=str)}]
        }


async def score_metrics_handler(args: dict[str, Any]) -> dict[str, Any]:
    """Compute composite score from pipeline metrics.

    Input:
      - metrics: dict with keys like auc, f1, train_time, explainability_coverage
      - weights: (optional) custom scoring weights

    Output:
      - composite_score: float
      - score_breakdown: str
      - per_component: dict of individual component scores
    """
    metrics = args["metrics"]
    weights = args.get("weights", None)

    if isinstance(metrics, str):
        try:
            metrics = json.loads(metrics)
        except json.JSONDecodeError:
            return {
                "content": [{"type": "text", "text": json.dumps({
                    "error": "metrics must be a valid JSON object",
                }, indent=2)}],
                "is_error": True,
            }

    w = weights or DEFAULT_WEIGHTS
    cs = composite_score(metrics, w)
    bd = score_breakdown(metrics, w)

    # Compute individual components
    auc = metrics.get("auc", 0.5)
    f1 = metrics.get("f1", 0.0)
    t = max(metrics.get("train_time", 999.0), 1.0)
    shap_cov = metrics.get("explainability_coverage", 0.0)
    llm_expl = metrics.get("llm_explainability_score", 0.5)
    speed = 1.0 / (1.0 + math.log1p(t / 30.0))

    output = {
        "composite_score": round(cs, 4),
        "score_breakdown": bd,
        "per_component": {
            "auc": {"value": round(auc, 4), "weight": w.get("auc", 0.45)},
            "f1": {"value": round(f1, 4), "weight": w.get("f1", 0.20)},
            "speed": {"value": round(speed, 4), "weight": w.get("speed", 0.10)},
            "shap_coverage": {"value": round(shap_cov, 4), "weight": w.get("shap_coverage", 0.10)},
            "llm_explainability": {"value": round(llm_expl, 4), "weight": w.get("llm_explainability", 0.15)},
        },
        "weights": w,
    }

    return {"content": [{"type": "text", "text": json.dumps(output, indent=2)}]}
