"""
config.py — single source of truth for all paths, constants, and hyperparameters.
Every other module imports from here. Nothing is hardcoded elsewhere.
"""

from pathlib import Path

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────
BASE_DIR    = Path(__file__).resolve().parent.parent   # loan/
DATA_DIR    = BASE_DIR / "data"
MODELS_DIR  = BASE_DIR / "models"

TRAIN_PATH  = DATA_DIR / "cs-training.csv"
TEST_PATH   = DATA_DIR / "cs-test.csv"

MODEL_PATH  = MODELS_DIR / "best_model.pkl"
FEATURES_PATH = MODELS_DIR / "feature_names.pkl"

# ─────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────
TARGET = "SeriousDlqin2yrs"

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

DELINQ_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
    "NumberOfTime60-89DaysPastDueNotWorse",
]

# ─────────────────────────────────────────────
# PREPROCESSING CONSTANTS
# ─────────────────────────────────────────────
DEBT_RATIO_CAP_PERCENTILE = 0.99   # cap DebtRatio at this percentile of train
UTIL_CAP                  = 1.0    # revolving utilization is [0, 1] by definition
DELINQ_SENTINEL           = 98     # value meaning "unknown" in delinquency cols

# ─────────────────────────────────────────────
# MODEL HYPERPARAMETERS
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE    = 0.2

XGBOOST_PARAMS = {
    "n_estimators":     500,
    "max_depth":        4,
    "learning_rate":    0.05,
    "subsample":        0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric":      "auc",
    "random_state":     RANDOM_STATE,
    "n_jobs":           -1,
}

LGBM_PARAMS = {
    "n_estimators":   500,
    "max_depth":      4,
    "learning_rate":  0.05,
    "subsample":      0.8,
    "colsample_bytree": 0.8,
    "random_state":   RANDOM_STATE,
    "n_jobs":         -1,
    "verbose":        -1,
}

LOGREG_PARAMS = {
    "max_iter":     1000,
    "random_state": RANDOM_STATE,
    "C":            0.1,
    "solver":       "lbfgs",
}
