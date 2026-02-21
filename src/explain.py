"""
Explainability layer — SHAP values for global and local interpretability.

Produces:
  • Global feature importance ranking
  • Per-prediction top-K contributing features
  • SHAP summary plot (saved to artifacts/)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shap

from src.config import (
    ARTIFACTS_DIR,
    FEATURE_NAMES_FILE,
    MODEL_FINAL_FILE,
    SHAP_VALUES_FILE,
)

logger = logging.getLogger(__name__)

TOP_K = 8


def compute_shap_values(
    X: np.ndarray,
    model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    save: bool = True,
) -> shap.Explanation:
    """Compute TreeExplainer SHAP values for the XGBoost model."""
    if model is None:
        model = joblib.load(ARTIFACTS_DIR / MODEL_FINAL_FILE)
    if feature_names is None:
        with open(ARTIFACTS_DIR / FEATURE_NAMES_FILE) as f:
            feature_names = json.load(f)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)
    shap_values.feature_names = feature_names

    if save:
        joblib.dump(shap_values, ARTIFACTS_DIR / SHAP_VALUES_FILE)
        logger.info("SHAP values saved → %s", ARTIFACTS_DIR / SHAP_VALUES_FILE)

    return shap_values


def global_importance(shap_values: shap.Explanation) -> List[Tuple[str, float]]:
    """Rank features by mean |SHAP| across the dataset."""
    mean_abs = np.abs(shap_values.values).mean(axis=0)
    names = shap_values.feature_names or [f"f{i}" for i in range(len(mean_abs))]
    ranking = sorted(zip(names, mean_abs), key=lambda x: x[1], reverse=True)
    return ranking


def local_explanation(
    shap_values: shap.Explanation,
    idx: int,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """Return the top-K SHAP contributors for a single prediction."""
    vals = shap_values.values[idx]
    names = shap_values.feature_names or [f"f{i}" for i in range(len(vals))]
    pairs = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return [
        {"feature": name, "shap_value": round(float(val), 4), "direction": "risk ↑" if val > 0 else "risk ↓"}
        for name, val in pairs
    ]


def plot_summary(shap_values: shap.Explanation, output_path: Optional[Path] = None) -> Path:
    """Generate and save a SHAP beeswarm summary plot."""
    output_path = output_path or ARTIFACTS_DIR / "shap_summary.png"
    fig, ax = plt.subplots(figsize=(10, 7))
    shap.plots.beeswarm(shap_values, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("SHAP summary plot → %s", output_path)
    return output_path


def explain_prediction(
    X_single: np.ndarray,
    model: Optional[Any] = None,
    feature_names: Optional[List[str]] = None,
    top_k: int = TOP_K,
) -> List[Dict[str, Any]]:
    """One-shot explanation for a single observation (used by the UI)."""
    if model is None:
        model = joblib.load(ARTIFACTS_DIR / MODEL_FINAL_FILE)
    if feature_names is None:
        with open(ARTIFACTS_DIR / FEATURE_NAMES_FILE) as f:
            feature_names = json.load(f)

    explainer = shap.TreeExplainer(model)
    sv = explainer(X_single)
    sv.feature_names = feature_names

    vals = sv.values[0]
    pairs = sorted(zip(feature_names, vals), key=lambda x: abs(x[1]), reverse=True)[:top_k]
    return [
        {"feature": name, "shap_value": round(float(val), 4), "direction": "risk ↑" if val > 0 else "risk ↓"}
        for name, val in pairs
    ]
