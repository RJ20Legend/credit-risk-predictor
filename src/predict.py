"""
predict.py — inference logic.

Loaded once at API startup. Accepts raw input, runs the full
preprocess → feature engineering → model pipeline, returns a probability.

Nothing in the API layer touches ML logic directly — it only calls predict().
"""

import pickle
import pandas as pd
from src.config     import MODEL_PATH, FEATURES_PATH
from src.preprocess import transform
from src.features   import build_features


def load_artifacts():
    """Load model, feature list, and preprocessor state from disk."""
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    with open(FEATURES_PATH, "rb") as f:
        feature_names = pickle.load(f)

    preprocessor_state_path = MODEL_PATH.parent / "preprocessor_state.pkl"
    with open(preprocessor_state_path, "rb") as f:
        state = pickle.load(f)

    return model, feature_names, state


def predict(input_data: dict, model, feature_names: list, state: dict) -> dict:
    """
    Run full inference pipeline on a single input.

    Args:
        input_data:    dict of raw feature values (same keys as RAW_FEATURES in config)
        model:         trained sklearn-compatible model
        feature_names: list of feature columns in the order the model expects
        state:         preprocessor state from fit_preprocessor()

    Returns:
        dict with 'probability' (float) and 'prediction' (0 or 1)
    """
    # Wrap input in a DataFrame (preprocess/features expect DataFrames)
    df = pd.DataFrame([input_data])

    # Apply the same pipeline used during training
    df_clean = transform(df, state, is_train=False)
    df_feat  = build_features(df_clean)

    # Select and order features exactly as during training
    X = df_feat[feature_names]

    prob       = float(model.predict_proba(X)[0, 1])
    prediction = int(prob >= 0.5)

    return {
        "probability": round(prob, 4),
        "prediction":  prediction,
        "risk_label":  "High Risk" if prediction == 1 else "Low Risk",
    }
