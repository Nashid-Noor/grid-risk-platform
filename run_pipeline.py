#!/usr/bin/env python3
"""
run_pipeline.py — Execute the full Phase 1 training + explainability pipeline.

Usage:
    python run_pipeline.py --data data/outage_data.csv

Produces all artifacts in artifacts/ and prints evaluation summary.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s — %(message)s")
logger = logging.getLogger("pipeline")


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid Risk Platform — full pipeline")
    parser.add_argument("--data", type=str, default=None, help="Path to outage CSV/XLSX")
    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. Data loading + target definition
    # ------------------------------------------------------------------
    logger.info("STEP 1 — Loading and preparing data")
    from src.data import get_dataset
    df = get_dataset(args.data)

    # ------------------------------------------------------------------
    # 2. Feature engineering + preprocessing
    # ------------------------------------------------------------------
    logger.info("STEP 2 — Feature engineering and train/test split")
    from src.features import prepare_splits
    X_train, X_test, y_train, y_test, preprocessor, feature_names = prepare_splits(df)

    # ------------------------------------------------------------------
    # 3. Model training
    # ------------------------------------------------------------------
    logger.info("STEP 3 — Training models")
    from src.train import train_baseline, train_xgb, _evaluate, _cross_validate, calibration_summary, save_drift_reference
    from src.config import ARTIFACTS_DIR, METRICS_FILE

    lr = train_baseline(X_train, y_train)
    lr_cv = _cross_validate(lr, X_train, y_train, "LR-baseline")
    lr_test = _evaluate(lr, X_test, y_test, "LR-baseline test")

    xgb = train_xgb(X_train, y_train)
    xgb_cv = _cross_validate(xgb, X_train, y_train, "XGB-final")
    xgb_test = _evaluate(xgb, X_test, y_test, "XGB-final test")

    calibration_summary(xgb, X_test, y_test)

    metrics = {
        "baseline": {**lr_test, "cv_roc_auc": lr_cv},
        "xgboost": {**xgb_test, "cv_roc_auc": xgb_cv},
    }
    with open(ARTIFACTS_DIR / METRICS_FILE, "w") as f:
        json.dump(metrics, f, indent=2)

    save_drift_reference(df)

    # ------------------------------------------------------------------
    # 4. Explainability
    # ------------------------------------------------------------------
    logger.info("STEP 4 — SHAP explainability")
    from src.explain import compute_shap_values, global_importance, local_explanation, plot_summary

    shap_values = compute_shap_values(X_test, model=xgb, feature_names=feature_names)
    ranking = global_importance(shap_values)
    plot_summary(shap_values)

    # ------------------------------------------------------------------
    # 5. Summary
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info("Metrics: %s", json.dumps(metrics, indent=2))
    logger.info("Top global features:")
    for feat, imp in ranking[:10]:
        logger.info("  %-35s  %.4f", feat, imp)

    sample_explanation = local_explanation(shap_values, idx=0)
    logger.info("Sample local explanation (row 0):")
    for entry in sample_explanation:
        logger.info("  %-35s  %+.4f  %s", entry["feature"], entry["shap_value"], entry["direction"])

    logger.info("All artifacts written to: %s", ARTIFACTS_DIR)


if __name__ == "__main__":
    main()
