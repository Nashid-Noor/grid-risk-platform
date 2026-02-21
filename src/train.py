"""
Training pipeline — baseline logistic regression + XGBoost final model.

Usage
-----
    python -m src.train                           # default data path
    python -m src.train --data path/to/outage.csv # custom path

Outputs (→ artifacts/)
    logistic_baseline.joblib   fitted LR model
    xgb_final.joblib           fitted XGBoost model
    preprocessor.joblib        sklearn ColumnTransformer
    metrics.json               evaluation results
    drift_reference.joblib     column-level stats for future drift checks
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

from src.config import (
    ARTIFACTS_DIR,
    DRIFT_REF_FILE,
    METRICS_FILE,
    MODEL_BASELINE_FILE,
    MODEL_FINAL_FILE,
    RANDOM_STATE,
)
from src.data import get_dataset
from src.features import engineer_features, prepare_splits, _resolve_columns

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger(__name__)

CV_FOLDS = 5


def _evaluate(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    label: str,
) -> Dict[str, float]:
    """Compute core classification metrics and return as dict."""
    y_prob = model.predict_proba(X)[:, 1]
    roc = roc_auc_score(y, y_prob)
    pr = average_precision_score(y, y_prob)
    brier = brier_score_loss(y, y_prob)

    logger.info("[%s]  ROC-AUC=%.4f  PR-AUC=%.4f  Brier=%.4f", label, roc, pr, brier)
    return {"roc_auc": round(roc, 4), "pr_auc": round(pr, 4), "brier": round(brier, 4)}


def _cross_validate(model: Any, X: np.ndarray, y: np.ndarray, label: str) -> float:
    """Stratified k-fold cross-validation on ROC-AUC."""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
    mean, std = scores.mean(), scores.std()
    logger.info("[%s CV]  ROC-AUC = %.4f ± %.4f", label, mean, std)
    return round(mean, 4)


def train_baseline(X_train: np.ndarray, y_train: np.ndarray) -> LogisticRegression:
    lr = LogisticRegression(
        class_weight="balanced",
        max_iter=1000,
        solver="lbfgs",
        random_state=RANDOM_STATE,
    )
    lr.fit(X_train, y_train)
    joblib.dump(lr, ARTIFACTS_DIR / MODEL_BASELINE_FILE)
    return lr


def train_xgb(X_train: np.ndarray, y_train: np.ndarray) -> XGBClassifier:
    """XGBoost with scale_pos_weight to handle class imbalance."""
    neg, pos = np.bincount(y_train)
    scale = neg / max(pos, 1)

    xgb = XGBClassifier(
        n_estimators=400,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale,
        eval_metric="aucpr",
        use_label_encoder=False,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)
    joblib.dump(xgb, ARTIFACTS_DIR / MODEL_FINAL_FILE)
    return xgb


def save_drift_reference(df_raw, path: Path = ARTIFACTS_DIR / DRIFT_REF_FILE) -> None:
    """
    Persist column-level statistics from training data so we can detect
    covariate drift at inference time.
    """
    df = engineer_features(df_raw)
    num_cols, _ = _resolve_columns(df)
    stats = {}
    for col in num_cols:
        if col in df.columns:
            s = df[col].dropna()
            stats[col] = {"mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max())}
    joblib.dump(stats, path)
    logger.info("Drift reference saved — %d features tracked", len(stats))


def calibration_summary(model: Any, X: np.ndarray, y: np.ndarray) -> None:
    """Log calibration quality — in production this feeds a monitoring dashboard."""
    y_prob = model.predict_proba(X)[:, 1]
    fraction_pos, mean_pred = calibration_curve(y, y_prob, n_bins=8, strategy="quantile")

    diffs = np.abs(fraction_pos - mean_pred)
    max_gap = diffs.max()
    logger.info(
        "Calibration check — max bin gap: %.3f %s",
        max_gap,
        "(acceptable)" if max_gap < 0.10 else "(consider Platt scaling)",
    )


def run(data_path: str | None = None) -> Dict[str, Any]:
    logger.info("=" * 60)
    logger.info("GRID RISK PLATFORM — Training pipeline")
    logger.info("=" * 60)

    df = get_dataset(data_path)
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_splits(df)

    # --- Baseline --------------------------------------------------------
    lr = train_baseline(X_train, y_train)
    lr_cv = _cross_validate(lr, X_train, y_train, "LR-baseline")
    lr_test = _evaluate(lr, X_test, y_test, "LR-baseline test")

    # --- XGBoost ---------------------------------------------------------
    xgb = train_xgb(X_train, y_train)
    xgb_cv = _cross_validate(xgb, X_train, y_train, "XGB-final")
    xgb_test = _evaluate(xgb, X_test, y_test, "XGB-final test")

    calibration_summary(xgb, X_test, y_test)

    # --- Persist metrics -------------------------------------------------
    metrics = {
        "baseline": {**lr_test, "cv_roc_auc": lr_cv},
        "xgboost": {**xgb_test, "cv_roc_auc": xgb_cv},
    }
    with open(ARTIFACTS_DIR / METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info("Metrics saved → %s", ARTIFACTS_DIR / METRICS_FILE)

    # --- Drift reference -------------------------------------------------
    save_drift_reference(df)

    logger.info("Training complete. Artifacts → %s", ARTIFACTS_DIR)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Grid Risk models")
    parser.add_argument("--data", type=str, default=None, help="Path to outage CSV/XLSX")
    args = parser.parse_args()
    run(data_path=args.data)


if __name__ == "__main__":
    # Allow running from project root: python -m src.train
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    main()
