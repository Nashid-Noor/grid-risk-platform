"""
Feature engineering and preprocessing pipeline.

Key design decisions
--------------------
* CUSTOMERS.AFFECTED and OUTAGE.DURATION are **removed** from the feature
  matrix because they directly define the target (data leakage).
* Weather is proxied via ANOMALY.LEVEL (Oceanic Niño Index) which is already
  in the dataset — no external API needed.
* Engineered features are derived solely from existing columns.
* The sklearn ColumnTransformer is serialised so inference uses the exact
  same transformations.
"""
from __future__ import annotations

import json
import logging
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    ARTIFACTS_DIR,
    CATEGORICAL_FEATURES,
    FEATURE_NAMES_FILE,
    LEAK_COLUMNS,
    NUMERIC_FEATURES,
    PREPROCESSOR_FILE,
    RANDOM_STATE,
    TARGET_COL,
    TEST_SIZE,
)

logger = logging.getLogger(__name__)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive additional columns from raw data — no external sources."""
    df = df.copy()

    # Demand-to-customer ratio: MW lost per thousand customers
    if "DEMAND.LOSS.MW" in df.columns and "TOTAL.CUSTOMERS" in df.columns:
        df["demand_per_1k_cust"] = (
            df["DEMAND.LOSS.MW"] / (df["TOTAL.CUSTOMERS"] / 1_000)
        ).replace([np.inf, -np.inf], np.nan)

    # Residential share of total sales (economic structure proxy)
    if "RES.SALES" in df.columns and "TOTAL.SALES" in df.columns:
        df["res_sales_share"] = (
            df["RES.SALES"] / df["TOTAL.SALES"]
        ).replace([np.inf, -np.inf], np.nan)

    # Price spread: industrial vs residential (rate design signal)
    if "RES.PRICE" in df.columns and "IND.PRICE" in df.columns:
        df["price_spread_res_ind"] = df["RES.PRICE"] - df["IND.PRICE"]

    # Population density contrast (urban vs rural stress indicator)
    if "POPDEN_URBAN" in df.columns and "POPDEN_RURAL" in df.columns:
        df["urban_rural_density_ratio"] = (
            df["POPDEN_URBAN"] / df["POPDEN_RURAL"].replace(0, np.nan)
        ).replace([np.inf, -np.inf], np.nan)

    # Season bucket from MONTH
    if "MONTH" in df.columns:
        month = df["MONTH"].astype(float)
        df["season"] = pd.cut(
            month,
            bins=[0, 3, 6, 9, 12],
            labels=["winter", "spring", "summer", "fall"],
            include_lowest=True,
        ).astype(str)

    return df


# Additional engineered numeric & categorical columns ----------------------
_ENGINEERED_NUMERIC = [
    "demand_per_1k_cust",
    "res_sales_share",
    "price_spread_res_ind",
    "urban_rural_density_ratio",
]
_ENGINEERED_CATEGORICAL = ["season"]


def _resolve_columns(df: pd.DataFrame) -> Tuple[list[str], list[str]]:
    """Return (numeric_cols, categorical_cols) present in df, minus leakage."""
    available = set(df.columns)
    leak = set(LEAK_COLUMNS)

    num = [c for c in NUMERIC_FEATURES + _ENGINEERED_NUMERIC if c in available and c not in leak]
    cat = [c for c in CATEGORICAL_FEATURES + _ENGINEERED_CATEGORICAL if c in available]
    return num, cat


def build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    """Construct a ColumnTransformer that handles imputation + encoding."""
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="MISSING")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, num_cols),
            ("cat", categorical_pipe, cat_cols),
        ],
        remainder="drop",
    )
    return preprocessor


def prepare_splits(
    df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, ColumnTransformer, list[str]]:
    """
    Full feature-engineering → split → fit-transform workflow.

    Returns
    -------
    X_train, X_test, y_train, y_test, fitted_preprocessor, feature_names
    """
    df = engineer_features(df)
    num_cols, cat_cols = _resolve_columns(df)
    logger.info("Feature columns — numeric: %d, categorical: %d", len(num_cols), len(cat_cols))

    X = df[num_cols + cat_cols]
    y = df[TARGET_COL].values

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    preprocessor = build_preprocessor(num_cols, cat_cols)
    X_train = preprocessor.fit_transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)

    # Recover human-readable feature names
    ohe = preprocessor.named_transformers_["cat"].named_steps["encoder"]
    cat_feature_names = list(ohe.get_feature_names_out(cat_cols))
    feature_names = num_cols + cat_feature_names

    # Persist preprocessor + feature names
    import joblib
    joblib.dump(preprocessor, ARTIFACTS_DIR / PREPROCESSOR_FILE)
    with open(ARTIFACTS_DIR / FEATURE_NAMES_FILE, "w") as f:
        json.dump(feature_names, f)

    logger.info("Train: %d samples | Test: %d samples", X_train.shape[0], X_test.shape[0])
    return X_train, X_test, y_train, y_test, preprocessor, feature_names
