"""
Credit Risk / Loan Default Prediction
======================================
Give Me Some Credit — Kaggle
EDA + Feature Engineering Pipeline

Run: python credit_risk_eda.py
Outputs: eda_report.txt, train_features.csv, test_features.csv
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
print("=" * 60)
print("STEP 1: LOADING DATA")
print("=" * 60)

train = pd.read_csv('/mnt/user-data/uploads/cs-training.csv', index_col=0)
test  = pd.read_csv('/mnt/user-data/uploads/cs-test.csv',     index_col=0)

print(f"Train shape: {train.shape}")
print(f"Test  shape: {test.shape}")

TARGET = 'SeriousDlqin2yrs'

# ─────────────────────────────────────────────
# 2. EDA — REAL INSIGHTS, NOT JUST HISTOGRAMS
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: EDA")
print("=" * 60)

# --- Class imbalance ---
default_rate = train[TARGET].mean()
print(f"\n[Imbalance] Default rate: {default_rate:.4f} ({default_rate*100:.2f}%)")
print(f"  Non-default: {(train[TARGET]==0).sum():,}")
print(f"  Default:     {(train[TARGET]==1).sum():,}")
print("  → Heavy imbalance (~6.7%). SMOTE + class weights needed.")

# --- Missing values ---
print("\n[Missing Values]")
missing = train.isnull().sum()
missing_pct = (missing / len(train) * 100).round(2)
for col in missing[missing > 0].index:
    print(f"  {col}: {missing[col]:,} ({missing_pct[col]}%)")
print("  → MonthlyIncome (19.8%) and NumberOfDependents (2.6%) need imputation.")

# --- Key outliers identified ---
print("\n[Outliers Identified]")
print(f"  RevolvingUtilization max: {train['RevolvingUtilizationOfUnsecuredLines'].max():.0f}  (expected: 0-1)")
print(f"  DebtRatio max:            {train['DebtRatio'].max():.0f}  (many > 1, some are clearly income-normalized)")
print(f"  Age == 0:                 {(train['age'] == 0).sum()} row(s)  → treat as missing")
print(f"  Delinquency columns with value 98: likely sentinel for 'unknown'")

# Confirm 98 as sentinel
for col in ['NumberOfTime30-59DaysPastDueNotWorse',
            'NumberOfTimes90DaysLate',
            'NumberOfTime60-89DaysPastDueNotWorse']:
    count_98 = (train[col] == 98).sum()
    if count_98 > 0:
        print(f"    {col}: {count_98} rows with value 98")

# --- Default rate by key features (the interesting EDA) ---
print("\n[Default Rate by Age Bucket]")
train['age_bucket'] = pd.cut(train['age'], bins=[0,25,35,45,55,65,120],
                              labels=['<25','25-35','35-45','45-55','55-65','65+'])
age_default = train.groupby('age_bucket', observed=True)[TARGET].mean().round(4)
for bucket, rate in age_default.items():
    bar = '█' * int(rate * 200)
    print(f"  {bucket:>6}: {rate:.4f}  {bar}")
train.drop('age_bucket', axis=1, inplace=True)

print("\n[Default Rate by Revolving Utilization Bucket]")
train['util_bucket'] = pd.cut(
    train['RevolvingUtilizationOfUnsecuredLines'].clip(0, 1),
    bins=[0, 0.3, 0.5, 0.75, 0.9, 1.0],
    labels=['0-30%','30-50%','50-75%','75-90%','90-100%']
)
util_default = train.groupby('util_bucket', observed=True)[TARGET].mean().round(4)
for bucket, rate in util_default.items():
    bar = '█' * int(rate * 100)
    print(f"  {bucket:>9}: {rate:.4f}  {bar}")
train.drop('util_bucket', axis=1, inplace=True)

print("\n[Default Rate by Number of Late Payments (30-59 days)]")
late_default = train.groupby('NumberOfTime30-59DaysPastDueNotWorse')[TARGET].mean()
for val, rate in late_default[late_default.index <= 5].items():
    bar = '█' * int(rate * 100)
    print(f"  {val} late payments: {rate:.4f}  {bar}")

# ─────────────────────────────────────────────
# 3. CLEANING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 3: CLEANING")
print("=" * 60)

def clean_data(df, is_train=True):
    df = df.copy()

    # Drop index column if present
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # Fix age=0 → NaN
    df['age'] = df['age'].replace(0, np.nan)

    # Clip revolving utilization to [0, 1] — values like 50708 are data errors
    df['RevolvingUtilizationOfUnsecuredLines'] = (
        df['RevolvingUtilizationOfUnsecuredLines'].clip(0, 1)
    )

    # Replace sentinel value 98 in delinquency columns with NaN
    delinq_cols = [
        'NumberOfTime30-59DaysPastDueNotWorse',
        'NumberOfTimes90DaysLate',
        'NumberOfTime60-89DaysPastDueNotWorse'
    ]
    for col in delinq_cols:
        df[col] = df[col].replace(98, np.nan)

    # Clip DebtRatio: extreme values are suspect; cap at 99th percentile
    debt_cap = df['DebtRatio'].quantile(0.99)
    df['DebtRatio'] = df['DebtRatio'].clip(0, debt_cap)

    return df

train = clean_data(train, is_train=True)
test  = clean_data(test,  is_train=False)
print("  Cleaning done.")

# ─────────────────────────────────────────────
# 4. IMPUTATION
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: IMPUTATION")
print("=" * 60)

# MonthlyIncome: median imputation grouped by NumberOfOpenCreditLinesAndLoans
# (richer imputation than a flat median — income correlates with credit activity)
income_medians = (
    train.groupby('NumberOfOpenCreditLinesAndLoans')['MonthlyIncome']
    .median()
)

def impute_income(df):
    df = df.copy()
    missing_mask = df['MonthlyIncome'].isnull()
    df.loc[missing_mask, 'MonthlyIncome'] = (
        df.loc[missing_mask, 'NumberOfOpenCreditLinesAndLoans']
        .map(income_medians)
        .fillna(train['MonthlyIncome'].median())  # fallback for unseen credit line counts
    )
    return df

train = impute_income(train)
test  = impute_income(test)

# NumberOfDependents: median imputation (simple, justified — no strong signal)
dep_median = train['NumberOfDependents'].median()
train['NumberOfDependents'] = train['NumberOfDependents'].fillna(dep_median)
test['NumberOfDependents']  = test['NumberOfDependents'].fillna(dep_median)

# Age: median imputation (only 1 row affected in train)
age_median = train['age'].median()
train['age'] = train['age'].fillna(age_median)
test['age']  = test['age'].fillna(age_median)

# Delinquency columns with NaN from 98-sentinel: fill with 0 (no known event = 0)
delinq_cols = [
    'NumberOfTime30-59DaysPastDueNotWorse',
    'NumberOfTimes90DaysLate',
    'NumberOfTime60-89DaysPastDueNotWorse'
]
for col in delinq_cols:
    train[col] = train[col].fillna(0)
    test[col]  = test[col].fillna(0)

print(f"  Remaining nulls — train: {train.isnull().sum().sum()}, test: {test.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 5. FEATURE ENGINEERING
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: FEATURE ENGINEERING")
print("=" * 60)

def engineer_features(df):
    df = df.copy()

    # --- Debt burden ---
    # Monthly debt payment estimate from DebtRatio × MonthlyIncome
    df['MonthlyDebtPayment'] = df['DebtRatio'] * df['MonthlyIncome']

    # Disposable income = income minus estimated debt payments
    df['DisposableIncome'] = df['MonthlyIncome'] - df['MonthlyDebtPayment']

    # Income per dependent (avoids division by zero)
    df['IncomePerDependent'] = df['MonthlyIncome'] / (df['NumberOfDependents'] + 1)

    # --- Delinquency severity score ---
    # Weighted sum: 90+ days late is far more serious than 30-59 days
    df['DelinquencyScore'] = (
        df['NumberOfTime30-59DaysPastDueNotWorse'] * 1 +
        df['NumberOfTime60-89DaysPastDueNotWorse'] * 2 +
        df['NumberOfTimes90DaysLate']              * 4
    )

    # Any delinquency flag (binary)
    df['EverDelinquent'] = (df['DelinquencyScore'] > 0).astype(int)

    # --- Credit utilization risk ---
    # High utilization is a strong default predictor
    df['HighUtilization'] = (
        df['RevolvingUtilizationOfUnsecuredLines'] > 0.75
    ).astype(int)

    # Utilization squared — captures non-linear risk at extreme values
    df['UtilizationSquared'] = df['RevolvingUtilizationOfUnsecuredLines'] ** 2

    # --- Age-related features ---
    # Young borrowers with high utilization = high risk segment
    df['YoungHighUtil'] = (
        (df['age'] < 35) &
        (df['RevolvingUtilizationOfUnsecuredLines'] > 0.75)
    ).astype(int)

    # Log age (diminishing marginal experience effect)
    df['LogAge'] = np.log1p(df['age'])

    # --- Real estate as stability proxy ---
    df['HasRealEstate'] = (df['NumberRealEstateLoansOrLines'] > 0).astype(int)

    # Total loans = real estate + other credit lines
    df['TotalCreditLines'] = (
        df['NumberRealEstateLoansOrLines'] +
        df['NumberOfOpenCreditLinesAndLoans']
    )

    # --- Income log-transform (heavy right skew) ---
    df['LogMonthlyIncome'] = np.log1p(df['MonthlyIncome'])

    # --- Debt ratio buckets (captures non-linearity) ---
    df['DebtRatioBucket'] = pd.cut(
        df['DebtRatio'],
        bins=[-np.inf, 0.3, 0.5, 0.75, 1.0, np.inf],
        labels=[0, 1, 2, 3, 4]
    ).astype(int)

    return df

train = engineer_features(train)
test  = engineer_features(test)

print(f"  New feature count: {len(train.columns) - 11} engineered features added")
print(f"  Total features (excl. target): {len(train.columns) - 1}")

# ─────────────────────────────────────────────
# 6. FEATURE → TARGET CORRELATION CHECK
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: FEATURE CORRELATIONS WITH TARGET")
print("=" * 60)

features = [c for c in train.columns if c != TARGET]
corr = train[features + [TARGET]].corr()[TARGET].drop(TARGET).sort_values(key=abs, ascending=False)

print("\n  Top 15 features by |correlation| with SeriousDlqin2yrs:")
for feat, val in corr.head(15).items():
    direction = '↑' if val > 0 else '↓'
    bar = '█' * int(abs(val) * 200)
    print(f"  {direction} {feat:<45} {val:+.4f}  {bar}")

# ─────────────────────────────────────────────
# 7. SAVE PROCESSED DATA
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 7: SAVING")
print("=" * 60)

train.to_csv('/home/claude/train_features.csv', index=False)
test.to_csv('/home/claude/test_features.csv',  index=False)

print(f"  train_features.csv: {train.shape}")
print(f"  test_features.csv:  {test.shape}")
print("\n✓ EDA + Feature Engineering complete.")
print("  Next step: model training (XGBoost / LightGBM / LogReg comparison)")
