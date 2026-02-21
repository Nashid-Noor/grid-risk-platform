"""
Inference module — load artifacts and score new observations.

Includes lightweight covariate drift detection that compares incoming
feature distributions against training-time reference statistics.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import numpy as np
import pandas as pd

from src.config import (
    ARTIFACTS_DIR,
    DRIFT_REF_FILE,
    FEATURE_NAMES_FILE,
    MODEL_FINAL_FILE,
    PREPROCESSOR_FILE,
)
from src.features import engineer_features, _resolve_columns

logger = logging.getLogger(__name__)

# Drift thresholds (z-score of column mean vs reference)
DRIFT_WARN_THRESHOLD = 2.0
DRIFT_ALERT_THRESHOLD = 3.5


class GridRiskPredictor:
    """Stateless predictor wrapping saved artifacts."""

    def __init__(self, artifacts_dir: Path = ARTIFACTS_DIR) -> None:
        self.model = joblib.load(artifacts_dir / MODEL_FINAL_FILE)
        self.preprocessor = joblib.load(artifacts_dir / PREPROCESSOR_FILE)
        with open(artifacts_dir / FEATURE_NAMES_FILE) as f:
            self.feature_names: List[str] = json.load(f)

        drift_path = artifacts_dir / DRIFT_REF_FILE
        self.drift_ref: Optional[Dict[str, Dict[str, float]]] = (
            joblib.load(drift_path) if drift_path.exists() else None
        )

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Score a DataFrame of raw outage records.

        Returns
        -------
        probabilities : np.ndarray  – P(high_impact)
        labels        : np.ndarray  – binary prediction at 0.5 threshold
        """
        df = engineer_features(df)
        X = self.preprocessor.transform(df)
        probs = self.model.predict_proba(X)[:, 1]
        labels = (probs >= 0.5).astype(int)
        return probs, labels

    def predict_single(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience wrapper for a single observation (used by UI)."""
        df = pd.DataFrame([record])
        probs, labels = self.predict(df)
        return {
            "probability": float(probs[0]),
            "prediction": int(labels[0]),
            "risk_tier": _risk_tier(probs[0]),
        }

    # ------------------------------------------------------------------
    # Drift detection
    # ------------------------------------------------------------------
    def check_drift(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Compare incoming batch column means against training reference.

        Returns a dict of {feature: status} where status ∈ {ok, warn, alert}.
        """
        if self.drift_ref is None:
            logger.warning("No drift reference found — skipping check.")
            return {}

        df = engineer_features(df)
        results: Dict[str, str] = {}
        for col, ref in self.drift_ref.items():
            if col not in df.columns:
                continue
            col_mean = df[col].dropna().mean()
            ref_mean, ref_std = ref["mean"], ref["std"]
            if ref_std == 0:
                continue
            z = abs(col_mean - ref_mean) / ref_std
            if z >= DRIFT_ALERT_THRESHOLD:
                status = "alert"
            elif z >= DRIFT_WARN_THRESHOLD:
                status = "warn"
            else:
                status = "ok"
            results[col] = status

        drifted = {k: v for k, v in results.items() if v != "ok"}
        if drifted:
            logger.warning("Drift detected: %s", drifted)
        return results


def _risk_tier(prob: float) -> str:
    if prob >= 0.75:
        return "CRITICAL"
    if prob >= 0.50:
        return "HIGH"
    if prob >= 0.25:
        return "MODERATE"
    return "LOW"
