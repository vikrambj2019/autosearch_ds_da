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

Available packages in ML venv:
  pandas, numpy, scipy, scikit-learn, lightgbm, xgboost, catboost,
  imbalanced-learn (imblearn), shap, optuna, feature-engine, tabulate, joblib
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
DROP_COLS = []  # Extra columns to drop (e.g., leaky features)
# ──────────────────────────────────────────────────────────────────────────────

# ── CONFIGURATION — Agent modifies this section ──────────────────────────────
MODEL_TYPE = "lgbm"          # lgbm | xgb | rf | catboost

RESAMPLE = "none"            # none | smote | adasyn | borderline_smote
RESAMPLE_RATIO = 0.5         # target minority/majority ratio after resampling

THRESHOLD_TUNING = False     # if True, sweep thresholds to maximize F1
BINARIZE_TARGET = False      # if True, convert target to binary (>0 → 1)

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

# CatBoost params
CATBOOST_PARAMS = dict(
    iterations=500,
    learning_rate=0.05,
    depth=6,
    auto_class_weights="Balanced",
)
# ──────────────────────────────────────────────────────────────────────────────


def load_and_preprocess(data_path, target_col):
    df = pd.read_csv(data_path)

    # Binarize target if configured
    if BINARIZE_TARGET:
        y = (df[target_col] > 0).astype(int)
    else:
        y = df[target_col].astype(int)

    # Auto-detect ID/entity columns to drop (uuid-like strings or unique-per-row)
    drop = list(DROP_COLS)
    for col in df.select_dtypes(include="object").columns:
        if col == target_col:
            continue
        if df[col].nunique() > 0.9 * len(df):
            drop.append(col)

    X = df.drop(columns=[target_col] + [c for c in drop if c in df.columns])

    # Drop constant columns (0 or 1 unique value)
    constant_cols = [c for c in X.columns if X[c].nunique() <= 1]
    if constant_cols:
        print(f"  Dropping {len(constant_cols)} constant columns")
        X = X.drop(columns=constant_cols)

    # Drop columns with >50% missing
    high_null = X.columns[X.isnull().mean() >= 0.5].tolist()
    if high_null:
        print(f"  Dropping {len(high_null)} columns with >50% nulls")
        X = X.drop(columns=high_null)

    # Encode categoricals
    le = LabelEncoder()
    for col in X.select_dtypes(include="object").columns:
        X[col] = le.fit_transform(X[col].astype(str).fillna("MISSING"))

    # Fill remaining nulls with median
    X = X.fillna(X.median(numeric_only=True))

    # Replace any remaining inf/-inf with NaN then fill
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    return X, y


def select_features(X_train, y_train, X_test):
    """Apply configured feature selection."""
    method = FEATURE_SELECTION

    if method == "none":
        return X_train, X_test, list(X_train.columns)

    elif method == "shap_top_k":
        import lightgbm as lgb
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


def resample_train(X_train, y_train):
    """Apply resampling to handle class imbalance. Always NaN-safe."""
    if RESAMPLE == "none":
        return X_train, y_train

    # Safety: ensure no NaN/inf before resampling (imblearn requires clean data)
    X_clean = X_train.fillna(0).replace([np.inf, -np.inf], 0)

    try:
        if RESAMPLE == "smote":
            from imblearn.over_sampling import SMOTE
            sampler = SMOTE(sampling_strategy=RESAMPLE_RATIO, random_state=42)
        elif RESAMPLE == "adasyn":
            from imblearn.over_sampling import ADASYN
            sampler = ADASYN(sampling_strategy=RESAMPLE_RATIO, random_state=42)
        elif RESAMPLE == "borderline_smote":
            from imblearn.over_sampling import BorderlineSMOTE
            sampler = BorderlineSMOTE(sampling_strategy=RESAMPLE_RATIO, random_state=42)
        else:
            print(f"  WARNING: Unknown RESAMPLE={RESAMPLE!r}, skipping")
            return X_train, y_train

        X_res, y_res = sampler.fit_resample(X_clean, y_train)
        X_res = pd.DataFrame(X_res, columns=X_train.columns)
        print(f"  Resampled: {len(X_train)} → {len(X_res)} (method={RESAMPLE})")
        return X_res, y_res

    except Exception as e:
        print(f"  WARNING: Resampling failed ({e}), using original data")
        return X_train, y_train


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

    elif model_type == "catboost":
        from catboost import CatBoostClassifier
        params = {**CATBOOST_PARAMS, "random_seed": 42, "verbose": 0}
        return CatBoostClassifier(**params)

    raise ValueError(f"Unknown MODEL_TYPE: {model_type!r}")


def compute_shap(model, X_sample, model_type):
    """Returns (n_samples, n_features) array of |SHAP values| for positive class."""
    try:
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
    except Exception as e:
        print(f"  WARNING: SHAP failed ({e}), using model feature_importances_")
        # Fallback: use model's built-in importance as a (1, n_features) array
        try:
            imp = model.feature_importances_
            return imp.reshape(1, -1)
        except Exception:
            return np.ones((1, X_sample.shape[1]))


def find_best_threshold(y_true, y_proba):
    """Sweep thresholds to find the one that maximizes F1."""
    best_f1, best_t = 0, 0.5
    for t in np.arange(0.1, 0.9, 0.02):
        preds = (y_proba >= t).astype(int)
        score = f1_score(y_true, preds, zero_division=0)
        if score > best_f1:
            best_f1, best_t = score, t
    return best_t, best_f1


def train_and_evaluate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos = float(neg / pos) if pos > 0 else 1.0
    print(f"  Class balance: {pos}/{neg} (ratio={scale_pos:.1f}:1)")

    # Feature selection
    X_train_s, X_test_s, selected_cols = select_features(X_train, y_train, X_test)

    # Resampling (applied AFTER feature selection, on clean data)
    X_train_r, y_train_r = resample_train(X_train_s, y_train)

    # Build and train model
    model = build_model(MODEL_TYPE, scale_pos)
    model.fit(X_train_r, y_train_r)

    # Predict
    y_proba = model.predict_proba(X_test_s)[:, 1]

    # Threshold tuning
    if THRESHOLD_TUNING:
        threshold, tuned_f1 = find_best_threshold(y_test, y_proba)
        print(f"  Tuned threshold: {threshold:.2f} (F1={tuned_f1:.4f})")
    else:
        threshold = 0.5

    y_pred = (y_proba >= threshold).astype(int)

    # Metrics
    try:
        auc = roc_auc_score(y_test, y_proba)
    except Exception:
        auc = 0.0

    f1 = f1_score(y_test, y_pred, zero_division=0)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)

    # SHAP explainability
    sample_n = min(500, len(X_test_s))
    X_sample = X_test_s.iloc[:sample_n]
    shap_arr = compute_shap(model, X_sample, MODEL_TYPE)

    mean_imp = shap_arr.mean(axis=0)
    total = mean_imp.sum()
    top10 = np.sort(mean_imp)[-10:].sum()
    explainability_coverage = float(top10 / total) if total > 0 else 0.0

    # Save top-30 SHAP features
    n_feats = min(30, len(selected_cols))
    top_idx = np.argsort(mean_imp)[-n_feats:][::-1]
    shap_features = [
        {"feature": selected_cols[i], "mean_abs_shap": float(mean_imp[i])}
        for i in top_idx
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
        "resample": RESAMPLE,
        "threshold": float(threshold),
        "feature_selection": FEATURE_SELECTION,
        "n_features": int(X_train.shape[1]),
        "n_selected_features": int(len(selected_cols)),
        "n_train": int(X_train_r.shape[0]),
        "n_test": int(X_test_s.shape[0]),
    }


if __name__ == "__main__":
    t0 = time.time()
    print(f"Model={MODEL_TYPE}  FeatureSelection={FEATURE_SELECTION}  Resample={RESAMPLE}")
    print(f"Loading: {DATA_PATH}")
    X, y = load_and_preprocess(DATA_PATH, TARGET_COL)
    print(f"  Shape={X.shape}  class_balance={y.mean():.4f}")

    print("Training...")
    metrics = train_and_evaluate(X, y)
    metrics["train_time"] = float(time.time() - t0)

    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print("\n=== METRICS ===")
    print(json.dumps(metrics, indent=2))
