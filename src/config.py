"""
Grid Risk Platform — central configuration.
All magic numbers, paths, and column mappings live here.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = ROOT_DIR / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)

RAW_DATA_PATH = ROOT_DIR / "data" / "outage_data.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.20

# ---------------------------------------------------------------------------
# Target definition
# An outage is "high-impact" when BOTH:
#   • CUSTOMERS.AFFECTED  >= 75th-percentile   AND
#   • OUTAGE.DURATION     >= 75th-percentile (minutes)
# This captures the top-severity tail that grid operators actually care about.
# ---------------------------------------------------------------------------
TARGET_COL = "high_impact"
CUSTOMERS_AFFECTED_COL = "CUSTOMERS.AFFECTED"
OUTAGE_DURATION_COL = "OUTAGE.DURATION"
IMPACT_QUANTILE = 0.75

# Columns expected from the Kaggle "Major Power Outage" dataset.
# The raw Excel/CSV typically has a few header rows to skip.
SKIP_ROWS_IN_RAW = 5          # adjust if your CSV export differs
SHEET_NAME = "Sheet1"          # only relevant when reading .xlsx

# Feature groups -----------------------------------------------------------
NUMERIC_FEATURES = [
    "ANOMALY.LEVEL",
    "OUTAGE.DURATION",          # dropped when it leaks into target; see features.py
    "DEMAND.LOSS.MW",
    "CUSTOMERS.AFFECTED",       # dropped when it leaks into target; see features.py
    "RES.PRICE",
    "COM.PRICE",
    "IND.PRICE",
    "TOTAL.PRICE",
    "TOTAL.SALES",
    "TOTAL.CUSTOMERS",
    "PC.REALGSP.STATE",
    "PC.REALGSP.REL",
    "PC.REALGSP.CHANGE",
    "UTIL.REALGSP",
    "UTIL.CONTRI",
    "POPULATION",
    "POPPCT_URBAN",
    "POPDEN_URBAN",
    "POPDEN_RURAL",
    "AREAPCT_URBAN",
    "PCT_LAND",
    "PCT_WATER_TOT",
]

CATEGORICAL_FEATURES = [
    "CLIMATE.REGION",
    "CLIMATE.CATEGORY",
    "CAUSE.CATEGORY",
    "NERC.REGION",
    "MONTH",
]

# Columns intentionally removed from features because they directly
# define the target or are high-cardinality identifiers.
LEAK_COLUMNS = [
    "CUSTOMERS.AFFECTED",
    "OUTAGE.DURATION",
]

# Artifact filenames -------------------------------------------------------
MODEL_BASELINE_FILE = "logistic_baseline.joblib"
MODEL_FINAL_FILE = "xgb_final.joblib"
PREPROCESSOR_FILE = "preprocessor.joblib"
METRICS_FILE = "metrics.json"
SHAP_VALUES_FILE = "shap_values.joblib"
FEATURE_NAMES_FILE = "feature_names.json"
DRIFT_REF_FILE = "drift_reference.joblib"
MONITORING_DIR = ROOT_DIR / "monitoring_reports"
MONITORING_DIR.mkdir(exist_ok=True)

# Model versioning
MODEL_VERSION = "1.0.0"
