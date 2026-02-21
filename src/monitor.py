"""
Monitoring module — Evidently-based drift detection and retrain triggers.

Usage
-----
    from src.monitor import DriftMonitor
    monitor = DriftMonitor(reference_df, current_df)
    report_path = monitor.generate_report()
    if monitor.should_retrain():
        # trigger retrain pipeline
"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
from evidently import ColumnMapping
from evidently.metric_preset import DataDriftPreset, TargetDriftPreset
from evidently.report import Report

from src.config import (
    CATEGORICAL_FEATURES,
    LEAK_COLUMNS,
    MONITORING_DIR,
    NUMERIC_FEATURES,
    TARGET_COL,
)
from src.features import engineer_features, _resolve_columns, _ENGINEERED_NUMERIC

logger = logging.getLogger(__name__)

# A retrain is suggested when this fraction of numeric features have drifted.
RETRAIN_DRIFT_FRACTION = 0.30


class DriftMonitor:
    """Compare a reference (training) dataset to a current (production) batch."""

    def __init__(
        self,
        reference_df: pd.DataFrame,
        current_df: pd.DataFrame,
        target_col: str = TARGET_COL,
    ) -> None:
        self.reference = engineer_features(reference_df.copy())
        self.current = engineer_features(current_df.copy())
        self.target_col = target_col

        num_cols, cat_cols = _resolve_columns(self.reference)
        # Remove leak columns that won't be present at inference time
        self.num_cols = [c for c in num_cols if c not in LEAK_COLUMNS]
        self.cat_cols = cat_cols

        self.column_mapping = ColumnMapping(
            target=self.target_col if self.target_col in self.reference.columns else None,
            numerical_features=self.num_cols,
            categorical_features=self.cat_cols,
        )

        self._report: Optional[Report] = None
        self._drift_summary: Optional[Dict] = None

    def generate_report(self, output_dir: Path = MONITORING_DIR) -> Path:
        """Run Evidently DataDriftPreset and save HTML + JSON reports."""
        metrics = [DataDriftPreset()]
        if self.target_col in self.reference.columns and self.target_col in self.current.columns:
            metrics.append(TargetDriftPreset())

        report = Report(metrics=metrics)
        report.run(
            reference_data=self.reference,
            current_data=self.current,
            column_mapping=self.column_mapping,
        )
        self._report = report

        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        html_path = output_dir / f"drift_report_{timestamp}.html"
        json_path = output_dir / f"drift_report_{timestamp}.json"

        report.save_html(str(html_path))
        report.save_json(str(json_path))

        # Parse the JSON for downstream decisions
        with open(json_path) as f:
            self._drift_summary = json.load(f)

        logger.info("Drift report saved → %s", html_path)
        return html_path

    def get_drifted_features(self) -> List[str]:
        """Return list of feature names flagged as drifted by Evidently."""
        if self._drift_summary is None:
            self.generate_report()

        drifted: List[str] = []
        try:
            metrics = self._drift_summary.get("metrics", [])
            for metric in metrics:
                result = metric.get("result", {})
                drift_by_columns = result.get("drift_by_columns", {})
                for col_name, col_info in drift_by_columns.items():
                    if col_info.get("drift_detected", False):
                        drifted.append(col_name)
        except (KeyError, TypeError):
            logger.warning("Could not parse drift summary — returning empty list.")
        return drifted

    def should_retrain(self) -> bool:
        """
        Simple retrain trigger: if more than RETRAIN_DRIFT_FRACTION of
        tracked numeric features have drifted, recommend retraining.
        """
        drifted = self.get_drifted_features()
        tracked = set(self.num_cols + self.cat_cols)
        drifted_tracked = [f for f in drifted if f in tracked]

        fraction = len(drifted_tracked) / max(len(tracked), 1)
        should = fraction >= RETRAIN_DRIFT_FRACTION

        logger.info(
            "Drift check — %d/%d features drifted (%.0f%%) → retrain=%s",
            len(drifted_tracked),
            len(tracked),
            fraction * 100,
            should,
        )
        return should


def retrain_trigger(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Tuple[bool, Path]:
    """
    Convenience function: run full monitoring pipeline and return
    (should_retrain, report_path).

    In production, this would be called by a scheduler (Airflow, cron, etc.)
    and the boolean would gate a retraining DAG.
    """
    monitor = DriftMonitor(reference_df, current_df)
    report_path = monitor.generate_report()
    return monitor.should_retrain(), report_path
