"""
Data ingestion and cleaning for the Major Power Outage dataset.

Assumptions
-----------
* Source: Kaggle "Major Power Outage Risks in the U.S." (Purdue / DOE).
* The CSV is pre-exported; if using the original .xlsx you may need to
  adjust `SKIP_ROWS_IN_RAW` in config.py.
* Rows with missing CUSTOMERS.AFFECTED *and* missing OUTAGE.DURATION are
  dropped because the target cannot be computed.
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    CUSTOMERS_AFFECTED_COL,
    IMPACT_QUANTILE,
    OUTAGE_DURATION_COL,
    RAW_DATA_PATH,
    SKIP_ROWS_IN_RAW,
    TARGET_COL,
)

logger = logging.getLogger(__name__)


def load_raw(path: Path | str | None = None, skip_rows: int = SKIP_ROWS_IN_RAW) -> pd.DataFrame:
    """Read the outage CSV into a DataFrame, handling common formatting issues."""
    path = Path(path) if path else RAW_DATA_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download from Kaggle and place it there."
        )

    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xls"):
        df = pd.read_excel(path, skiprows=skip_rows)
    else:
        # Try reading with skiprows first; fall back to 0 if it blows up.
        try:
            df = pd.read_csv(path, skiprows=skip_rows)
        except Exception:
            df = pd.read_csv(path)

    # Strip whitespace from column names
    df.columns = df.columns.str.strip()

    # Drop fully empty rows / columns (common artifact of Excel exports)
    df.dropna(how="all", axis=0, inplace=True)
    df.dropna(how="all", axis=1, inplace=True)

    logger.info("Loaded %d rows, %d columns from %s", len(df), len(df.columns), path)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Basic type coercion and null handling."""
    df = df.copy()
    
    # The first row often contains units (e.g. "Megawatt", "mins"). Drop it!
    if df["YEAR"].iloc[0] == "Year" or df["ANOMALY.LEVEL"].iloc[0] == "numeric":
        df = df.iloc[1:].reset_index(drop=True)

    # Coerce numeric columns that may have been read as object
    from src.config import NUMERIC_FEATURES
    
    # We also include extra columns used for feature engineering that might not be in NUMERIC_FEATURES
    num_cols = list(set(NUMERIC_FEATURES + [
        CUSTOMERS_AFFECTED_COL, OUTAGE_DURATION_COL, "DEMAND.LOSS.MW",
        "TOTAL.CUSTOMERS", "RES.SALES", "TOTAL.SALES", "RES.PRICE", "IND.PRICE",
        "POPDEN_URBAN", "POPDEN_RURAL"
    ]))
    for col in num_cols:
        if col in df.columns:
            if df[col].dtype == object and df[col].astype(str).str.contains(',').any():
                df[col] = df[col].astype(str).str.replace(',', '')
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Parse MONTH as string to avoid scikit-learn imputer mixed-type errors
    if "MONTH" in df.columns:
        df["MONTH"] = pd.to_numeric(df["MONTH"], errors="coerce").fillna(-1).astype(int).astype(str)
        df.loc[df["MONTH"] == "-1", "MONTH"] = np.nan

    # Drop rows where we cannot define the target at all
    target_deps = [c for c in [CUSTOMERS_AFFECTED_COL, OUTAGE_DURATION_COL] if c in df.columns]
    df.dropna(subset=target_deps, how="all", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("After cleaning: %d rows remain", len(df))
    return df


def build_target(df: pd.DataFrame, quantile: float = IMPACT_QUANTILE) -> pd.DataFrame:
    """
    Define binary target: **high_impact**.

    An outage qualifies as high-impact when:
      customers_affected >= Q(quantile)  AND  duration >= Q(quantile)

    This deliberately selects the upper-right quadrant of severity, which
    is the segment an operations center would triage first.

    When either column is missing for a row, we fall back to the single
    available column so we don't lose too many samples.
    """
    df = df.copy()

    cust_thresh = df[CUSTOMERS_AFFECTED_COL].quantile(quantile)
    dur_thresh = df[OUTAGE_DURATION_COL].quantile(quantile)

    has_cust = df[CUSTOMERS_AFFECTED_COL].notna()
    has_dur = df[OUTAGE_DURATION_COL].notna()

    high_cust = df[CUSTOMERS_AFFECTED_COL] >= cust_thresh
    high_dur = df[OUTAGE_DURATION_COL] >= dur_thresh

    # Both present → require both; only one present → use that one
    target = pd.Series(0, index=df.index)
    both = has_cust & has_dur
    target.loc[both] = ((high_cust & high_dur) & both).astype(int).loc[both]
    target.loc[has_cust & ~has_dur] = high_cust.astype(int).loc[has_cust & ~has_dur]
    target.loc[~has_cust & has_dur] = high_dur.astype(int).loc[~has_cust & has_dur]

    df[TARGET_COL] = target.astype(int)

    pos_rate = df[TARGET_COL].mean()
    logger.info(
        "Target built — positive rate: %.2f%% (%d / %d) | thresholds: customers>=%.0f, duration>=%.0f min",
        pos_rate * 100,
        df[TARGET_COL].sum(),
        len(df),
        cust_thresh,
        dur_thresh,
    )
    return df


def get_dataset(path: Path | str | None = None) -> pd.DataFrame:
    """End-to-end: load → clean → build target."""
    df = load_raw(path)
    df = clean(df)
    df = build_target(df)
    return df
