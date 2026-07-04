"""
preprocess.py — cleaning and imputation.

Two public functions:
    fit_preprocessor(train_df)  → returns a fitted PreprocessorState (a plain dict)
    transform(df, state)        → returns a cleaned copy of df

The "state" dict holds values computed on train (medians, caps) so the
exact same transforms are applied to test and to live API requests —
no data leakage, no silent differences.
"""

import numpy as np
import pandas as pd
from src.config import (
    TARGET, RAW_FEATURES, DELINQ_COLS,
    DEBT_RATIO_CAP_PERCENTILE, UTIL_CAP, DELINQ_SENTINEL,
)


def fit_preprocessor(train_df: pd.DataFrame) -> dict:
    """
    Learn all imputation values from training data only.
    Call this once on train, then pass the returned state to transform().
    """
    df = train_df.copy()

    # Debt ratio cap — learned from train
    debt_ratio_cap = df["DebtRatio"].quantile(DEBT_RATIO_CAP_PERCENTILE)

    # Income median per credit-line bucket (richer than flat median)
    df_clean_income = df.copy()
    df_clean_income["MonthlyIncome"] = df_clean_income["MonthlyIncome"].replace(0, np.nan)
    income_medians_by_credit = (
        df_clean_income
        .groupby("NumberOfOpenCreditLinesAndLoans")["MonthlyIncome"]
        .median()
        .to_dict()
    )
    income_global_median = df_clean_income["MonthlyIncome"].median()

    # Other simple medians
    dep_median = df["NumberOfDependents"].median()
    age_median = df["age"].replace(0, np.nan).median()

    state = {
        "debt_ratio_cap":           debt_ratio_cap,
        "income_medians_by_credit": income_medians_by_credit,
        "income_global_median":     income_global_median,
        "dep_median":               dep_median,
        "age_median":               age_median,
    }
    return state


def transform(df: pd.DataFrame, state: dict, is_train: bool = False) -> pd.DataFrame:
    """
    Apply cleaning + imputation to any dataframe using values from state.
    Set is_train=True when processing the training set (keeps TARGET column).
    """
    df = df.copy()

    # Keep only the columns we actually need
    cols_to_keep = RAW_FEATURES.copy()
    if is_train and TARGET in df.columns:
        cols_to_keep = [TARGET] + cols_to_keep
    df = df[cols_to_keep]

    # ── Fix age = 0 (data error, treat as missing) ──────────────────────
    df["age"] = df["age"].replace(0, np.nan)
    df["age"] = df["age"].fillna(state["age_median"])

    # ── Clip revolving utilization to [0, 1] ────────────────────────────
    # Values like 50,708 are data entry errors — credit utilization is a ratio
    df["RevolvingUtilizationOfUnsecuredLines"] = (
        df["RevolvingUtilizationOfUnsecuredLines"].clip(0, UTIL_CAP)
    )

    # ── Replace sentinel value 98 in delinquency cols with NaN → 0 ──────
    # 98 means "unknown number of late payments", not actually 98 times late
    for col in DELINQ_COLS:
        df[col] = df[col].replace(DELINQ_SENTINEL, np.nan).fillna(0)

    # ── Cap DebtRatio at 99th percentile (extreme values are noise) ──────
    df["DebtRatio"] = df["DebtRatio"].clip(0, state["debt_ratio_cap"])

    # ── Impute MonthlyIncome using per-credit-line-bucket median ─────────
    missing_income = df["MonthlyIncome"].isnull()
    df.loc[missing_income, "MonthlyIncome"] = (
        df.loc[missing_income, "NumberOfOpenCreditLinesAndLoans"]
        .map(state["income_medians_by_credit"])
        .fillna(state["income_global_median"])   # fallback for unseen bucket values
    )

    # ── Impute NumberOfDependents ─────────────────────────────────────────
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(state["dep_median"])

    return df
