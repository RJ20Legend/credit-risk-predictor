"""
features.py — feature engineering.

One public function:
    build_features(df) → returns df with engineered columns added

All transforms are deterministic (no fit needed) so this can be called
identically on train, test, and live API requests.
"""

import numpy as np
import pandas as pd


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to a cleaned dataframe.
    Input df must have already been through preprocess.transform().
    """
    df = df.copy()

    # ── Delinquency features ─────────────────────────────────────────────
    # Weighted severity score: 90+ days late is ~4x worse than 30-59 days
    df["DelinquencyScore"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"] * 1 +
        df["NumberOfTime60-89DaysPastDueNotWorse"] * 2 +
        df["NumberOfTimes90DaysLate"]              * 4
    )

    # Binary: has this person ever been late on a payment?
    # Turned out to be the single highest-correlated feature with default (+0.31)
    df["EverDelinquent"] = (df["DelinquencyScore"] > 0).astype(int)

    # ── Credit utilization features ──────────────────────────────────────
    # Default rate jumps from 2% at <30% util to 22% at >90% util
    df["HighUtilization"] = (
        df["RevolvingUtilizationOfUnsecuredLines"] > 0.75
    ).astype(int)

    # Squared utilization: captures the non-linear explosion in risk at extremes
    df["UtilizationSquared"] = df["RevolvingUtilizationOfUnsecuredLines"] ** 2

    # ── Debt burden features ─────────────────────────────────────────────
    # Reconstruct estimated monthly debt payment from DebtRatio × Income
    df["MonthlyDebtPayment"] = df["DebtRatio"] * df["MonthlyIncome"]

    # What's left after debt payments — negative = underwater
    df["DisposableIncome"] = df["MonthlyIncome"] - df["MonthlyDebtPayment"]

    # Income adjusted for number of dependents (+1 avoids division by zero)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # ── Age features ─────────────────────────────────────────────────────
    # Log age: experience effect is non-linear (diminishing returns)
    df["LogAge"] = np.log1p(df["age"])

    # Highest-risk segment: young borrowers who are already maxed out
    df["YoungHighUtil"] = (
        (df["age"] < 35) &
        (df["RevolvingUtilizationOfUnsecuredLines"] > 0.75)
    ).astype(int)

    # ── Credit line features ─────────────────────────────────────────────
    # Real estate ownership as a stability/wealth proxy
    df["HasRealEstate"] = (df["NumberRealEstateLoansOrLines"] > 0).astype(int)

    # Total credit exposure
    df["TotalCreditLines"] = (
        df["NumberRealEstateLoansOrLines"] +
        df["NumberOfOpenCreditLinesAndLoans"]
    )

    # ── Log-transform skewed features ───────────────────────────────────
    # MonthlyIncome has heavy right skew — log normalises the distribution
    df["LogMonthlyIncome"] = np.log1p(df["MonthlyIncome"])

    # ── Debt ratio bucket ─────────────────────────────────────────────────
    # Ordinal encoding of debt ratio bands (non-linear risk profile)
    df["DebtRatioBucket"] = pd.cut(
        df["DebtRatio"],
        bins=[-np.inf, 0.3, 0.5, 0.75, 1.0, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    return df


def get_feature_names(df: pd.DataFrame, target: str) -> list:
    """Return list of feature column names (everything except target)."""
    return [c for c in df.columns if c != target]
