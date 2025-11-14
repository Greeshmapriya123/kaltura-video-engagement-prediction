import joblib
import pandas as pd
import numpy as np

# Load preprocessor + models
preprocessor = joblib.load("models/preprocessor.pkl")
model_completion = joblib.load("models/xgb_completion.pkl")
model_dropoff = joblib.load("models/xgb_dropoff.pkl")

def run_prediction(df):
    """
    df: raw uploaded dataframe from user
    returns: dataframe with predictions
    """

    # 1️⃣ Select the exact same features used during training
    feature_cols = [
        "media_type", "duration_msecs", "count_plays", "sum_time_viewed",
        "avg_time_viewed", "count_loads", "load_play_ratio",
        "unique_known_users", "engagement_ranking"
    ]

    # Check missing columns
    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[feature_cols]

    # 2️⃣ Transform through preprocessor
    X_processed = preprocessor.transform(X)

    # 3️⃣ Predict using both models
    pred_completion = model_completion.predict(X_processed)
    pred_dropoff = model_dropoff.predict(X_processed)

    # 4️⃣ Combine output
    result = df.copy()
    result["predicted_completion_rate"] = np.round(pred_completion, 2)
    result["predicted_dropoff_risk"] = np.round(pred_dropoff, 2)

    return result
