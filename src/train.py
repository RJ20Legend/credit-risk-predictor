"""
train.py — model training, comparison, and saving the best model.

Run directly:
    python src/train.py

What it does:
    1. Loads and preprocesses data
    2. Engineers features
    3. Splits train / validation
    4. Trains XGBoost, LightGBM, Logistic Regression
    5. Handles class imbalance via scale_pos_weight / class_weight
    6. Evaluates on ROC-AUC and Precision-Recall AUC
    7. Saves the best model + feature names to models/
"""

import pickle
import numpy as np
import pandas as pd

from sklearn.model_selection  import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model     import LogisticRegression
from sklearn.preprocessing    import StandardScaler
from sklearn.pipeline         import Pipeline
from sklearn.metrics          import (
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
)
from xgboost  import XGBClassifier
from lightgbm import LGBMClassifier

from src.config     import (
    TRAIN_PATH, MODEL_PATH, FEATURES_PATH, TARGET,
    RANDOM_STATE, TEST_SIZE,
    XGBOOST_PARAMS, LGBM_PARAMS, LOGREG_PARAMS,
)
from src.preprocess import fit_preprocessor, transform
from src.features   import build_features, get_feature_names


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def evaluate(name: str, model, X_val: pd.DataFrame, y_val: pd.Series) -> dict:
    """Print and return evaluation metrics for a trained model."""
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = (y_prob >= 0.5).astype(int)

    roc_auc = roc_auc_score(y_val, y_prob)
    pr_auc  = average_precision_score(y_val, y_prob)

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  ROC-AUC : {roc_auc:.4f}")
    print(f"  PR-AUC  : {pr_auc:.4f}  ← key metric for imbalanced data")
    print(f"\n  Classification Report (threshold=0.5):")
    print(classification_report(y_val, y_pred, target_names=["No Default", "Default"]))

    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    print(f"  Confusion Matrix:")
    print(f"    TP={tp:,}  FP={fp:,}")
    print(f"    FN={fn:,}  TN={tn:,}")

    return {"name": name, "model": model, "roc_auc": roc_auc, "pr_auc": pr_auc}


# ─────────────────────────────────────────────
# MAIN TRAINING PIPELINE
# ─────────────────────────────────────────────

def main():
    print("=" * 60)
    print("CREDIT RISK — MODEL TRAINING")
    print("=" * 60)

    # ── 1. Load ──────────────────────────────────────────────────────────
    print("\n[1/6] Loading data...")
    raw_train = pd.read_csv(TRAIN_PATH, index_col=0)
    print(f"  Raw train shape: {raw_train.shape}")

    # ── 2. Preprocess ────────────────────────────────────────────────────
    print("\n[2/6] Preprocessing...")
    state = fit_preprocessor(raw_train)
    train_clean = transform(raw_train, state, is_train=True)
    print(f"  Clean train shape: {train_clean.shape}")
    print(f"  Nulls remaining: {train_clean.isnull().sum().sum()}")

    # ── 3. Feature engineering ───────────────────────────────────────────
    print("\n[3/6] Engineering features...")
    train_feat = build_features(train_clean)
    feature_names = get_feature_names(train_feat, TARGET)
    print(f"  Total features: {len(feature_names)}")

    X = train_feat[feature_names]
    y = train_feat[TARGET]

    # ── 4. Train/val split ───────────────────────────────────────────────
    print("\n[4/6] Splitting train/validation...")
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,   # preserve class ratio in both splits
    )
    print(f"  Train: {X_train.shape[0]:,}  |  Val: {X_val.shape[0]:,}")
    print(f"  Default rate — train: {y_train.mean():.4f}  val: {y_val.mean():.4f}")

    # ── 5. Handle class imbalance ────────────────────────────────────────
    # Scale_pos_weight = ratio of negatives to positives
    # This tells tree models to penalise missing a default more than a false alarm
    neg  = (y_train == 0).sum()
    pos  = (y_train == 1).sum()
    spw  = neg / pos
    print(f"\n  Class imbalance ratio (scale_pos_weight): {spw:.2f}")

    # ── 6. Train models ──────────────────────────────────────────────────
    print("\n[5/6] Training models...")

    # --- XGBoost ---
    xgb_params = {**XGBOOST_PARAMS, "scale_pos_weight": spw}
    xgb_params.pop("use_label_encoder", None)   # deprecated in newer xgboost
    xgb = XGBClassifier(**xgb_params)
    xgb.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    # --- LightGBM ---
    lgbm_params = {**LGBM_PARAMS, "scale_pos_weight": spw}
    lgbm = LGBMClassifier(**lgbm_params)
    lgbm.fit(X_train, y_train)

    # --- Logistic Regression (needs scaling) ---
    logreg = Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(**LOGREG_PARAMS, class_weight="balanced")),
    ])
    logreg.fit(X_train, y_train)

    # ── 7. Evaluate ──────────────────────────────────────────────────────
    print("\n[6/6] Evaluation on validation set:")
    results = [
        evaluate("XGBoost",             xgb,   X_val, y_val),
        evaluate("LightGBM",            lgbm,  X_val, y_val),
        evaluate("Logistic Regression", logreg, X_val, y_val),
    ]

    # ── 8. Pick best model by ROC-AUC ────────────────────────────────────
    best = max(results, key=lambda r: r["roc_auc"])
    print(f"\n{'='*60}")
    print(f"  Best model: {best['name']}")
    print(f"  ROC-AUC:    {best['roc_auc']:.4f}")
    print(f"  PR-AUC:     {best['pr_auc']:.4f}")
    print(f"{'='*60}")

    # ── 9. Save ──────────────────────────────────────────────────────────
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(best["model"], f)

    with open(FEATURES_PATH, "wb") as f:
        pickle.dump(feature_names, f)

    # Also save the preprocessor state for use in the API
    with open(MODEL_PATH.parent / "preprocessor_state.pkl", "wb") as f:
        pickle.dump(state, f)

    print(f"\n  Saved: {MODEL_PATH}")
    print(f"  Saved: {FEATURES_PATH}")
    print(f"  Saved: {MODEL_PATH.parent / 'preprocessor_state.pkl'}")
    print("\n✓ Training complete. Next: run the API with `uvicorn api.main:app --reload`")


if __name__ == "__main__":
    main()
