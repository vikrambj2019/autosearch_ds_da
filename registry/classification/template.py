#!/usr/bin/env python3
# STRATEGY: baseline — LightGBM binary classification with SHAP explainability
"""
Binary classification pipeline template.
The ML builder agent modifies the CONFIGURATION section to customize.

Outputs:
  metrics.json       — AUC, F1, accuracy, precision, recall, train_time, etc.
  shap_features.json — top-30 features with mean_abs_shap

DO NOT CHANGE: DATA_PATH, TARGET_COL, METRICS_OUT, SHAP_OUT
(these are injected by the run_pipeline tool)
"""
import json
import time
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    precision_score, recall_score,
)

# ── Fixed contract points (injected by run_pipeline tool) ─────────────────────
DATA_PATH = "PLACEHOLDER"
TARGET_COL = "PLACEHOLDER"
METRICS_OUT = Path("metrics.json")
SHAP_OUT = Path("shap_features.json")
DROP_COLS = ["FEATURE_STORE_MEMBER_ID"]
# ──────────────────────────────────────────────────────────────────────────────

# ── CONFIGURATION — Agent modifies this section ──────────────────────────────
MODEL_TYPE = "lgbm"          # lgbm | xgb | rf

FEATURE_SELECTION = "none"   # none | shap_top_k | correlation | mutual_info | variance
FEATURE_SELECTION_K = 80     # for shap_top_k / mutual_info: number of features to keep
CORR_THRESHOLD = 0.95        # for correlation: drop one of each pair above this

# LightGBM params
LGBM_PARAMS = dict(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    min_child_samples=40,
    colsample_bytree=0.8,
    subsample=0.8,
    subsample_freq=1,
    reg_alpha=0.1,
    reg_lambda=0.1,
    scale_pos_weight=None,   # set at runtime from class balance
)

# XGBoost params
XGB_PARAMS = dict(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    scale_pos_weight=None,   # set at runtime
)

# RandomForest params
RF_PARAMS = dict(
    n_estimators=200,
    max_depth=10,
    min_samples_leaf=20,
    max_features="sqrt",
    class_weight="balanced",
)
# ──────────────────────────────────────────────────────────────────────────────


def load_and_preprocess(data_path, target_col):
    df = pd.read_csv(data_path)
    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col] + [c for c in DROP_COLS if c in df.columns])

    # Drop columns with >50% missing
    X = X.loc[:, X.isnull().mean() < 0.5]

    # Encode categoricals
    le = LabelEncoder()
    for col in X.select_dtypes(include="object").columns:
        X[col] = le.fit_transform(X[col].astype(str).fillna("MISSING"))

    # Fill remaining nulls
    X = X.fillna(X.median(numeric_only=True))
    return X, y


def select_features(X_train, y_train, X_test):
    """Apply configured feature selection."""
    import lightgbm as lgb

    method = FEATURE_SELECTION

    if method == "none":
        return X_train, X_test, list(X_train.columns)

    elif method == "shap_top_k":
        proxy = lgb.LGBMClassifier(n_estimators=50, random_state=42, n_jobs=-1, verbose=-1)
        proxy.fit(X_train, y_train)
        imp = proxy.feature_importances_
        top_idx = np.argsort(imp)[-FEATURE_SELECTION_K:]
        cols = list(X_train.columns[top_idx])

    elif method == "correlation":
        corr = X_train.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        to_drop = {col for col in upper.columns if (upper[col] > CORR_THRESHOLD).any()}
        cols = [c for c in X_train.columns if c not in to_drop]

    elif method == "mutual_info":
        from sklearn.feature_selection import mutual_info_classif
        mi = mutual_info_classif(X_train, y_train, random_state=42, n_neighbors=5)
        top_idx = np.argsort(mi)[-FEATURE_SELECTION_K:]
        cols = list(X_train.columns[top_idx])

    elif method == "variance":
        from sklearn.feature_selection import VarianceThreshold
        sel = VarianceThreshold(threshold=0.01)
        sel.fit(X_train)
        cols = list(X_train.columns[sel.get_support()])

    else:
        cols = list(X_train.columns)

    return X_train[cols], X_test[cols], cols


def build_model(model_type, scale_pos):
    if model_type == "lgbm":
        import lightgbm as lgb
        params = {**LGBM_PARAMS, "scale_pos_weight": scale_pos,
                  "random_state": 42, "n_jobs": -1, "verbose": -1}
        return lgb.LGBMClassifier(**params)

    elif model_type == "xgb":
        import xgboost as xgb
        params = {**XGB_PARAMS, "scale_pos_weight": scale_pos,
                  "random_state": 42, "n_jobs": -1, "verbosity": 0}
        return xgb.XGBClassifier(**params)

    elif model_type == "rf":
        from sklearn.ensemble import RandomForestClassifier
        params = {**RF_PARAMS, "random_state": 42, "n_jobs": -1}
        return RandomForestClassifier(**params)

    raise ValueError(f"Unknown MODEL_TYPE: {model_type!r}")


def compute_shap(model, X_sample, model_type):
    """Returns (n_samples, n_features) array of |SHAP values| for positive class."""
    import shap
    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_sample)

    if isinstance(sv, list):
        arr = np.abs(sv[1]) if len(sv) >= 2 else np.abs(sv[0])
    elif sv.ndim == 3:
        arr = np.abs(sv[:, :, 1])
    else:
        arr = np.abs(sv)
    return arr


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = float(neg / pos) if pos > 0 else 1.0

    X_train_s, X_test_s, selected_cols = select_features(X_train, y_train, X_test)

    model = build_model(MODEL_TYPE, scale_pos)
    model.fit(X_train_s, y_train)

    y_proba = model.predict_proba(X_test_s)[:, 1]
    y_pred = model.predict(X_test_s)

    auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    # SHAP explainability
    sample_n = min(500, len(X_test_s))
    X_sample = X_test_s.iloc[:sample_n]
    shap_arr = compute_shap(model, X_sample, MODEL_TYPE)

    mean_imp = shap_arr.mean(axis=0)
    total = mean_imp.sum()
    top10 = np.sort(mean_imp)[-10:].sum()
    explainability_coverage = float(top10 / total) if total > 0 else 0.0

    # Save top-30 SHAP features
    top30_idx = np.argsort(mean_imp)[-30:][::-1]
    shap_features = [
        {"feature": selected_cols[i], "mean_abs_shap": float(mean_imp[i])}
        for i in top30_idx
    ]
    SHAP_OUT.write_text(json.dumps(shap_features, indent=2))

    return {
        "auc": float(auc),
        "f1": float(f1),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "explainability_coverage": explainability_coverage,
        "model_type": MODEL_TYPE,
        "feature_selection": FEATURE_SELECTION,
        "n_features": int(X_train.shape[1]),
        "n_selected_features": int(len(selected_cols)),
        "n_train": int(X_train_s.shape[0]),
        "n_test": int(X_test_s.shape[0]),
    }


if __name__ == "__main__":
    t0 = time.time()
    print(f"Model={MODEL_TYPE}  FeatureSelection={FEATURE_SELECTION}")
    print(f"Loading: {DATA_PATH}")
    X, y = load_and_preprocess(DATA_PATH, TARGET_COL)
    print(f"  Shape={X.shape}  class_balance={y.mean():.4f}")

    print("Training...")
    metrics = train_and_evaluate(X, y)
    metrics["train_time"] = float(time.time() - t0)

    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
