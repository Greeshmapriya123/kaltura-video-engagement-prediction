import pandas as pd
import joblib
import numpy as np

# -------------------------------------------------------
# Load all saved model artifacts
# -------------------------------------------------------
preprocessor = joblib.load("models/preprocessor.pkl")
model_completion = joblib.load("models/xgb_completion.pkl")
model_dropoff = joblib.load("models/xgb_dropoff.pkl")

# These are the ONLY 13 features the model was trained on
TRAIN_FEATURES = [
    "minutes",
    "media_type",
    "entry_source",
    "count_plays",
    "count_loads",
    "load_play_ratio",
    "sum_time_viewed",
    "avg_time_viewed",
    "avg_view_drop_off",
    "unique_known_users",
    "engagement_ranking",
    "avg_completion_rate",
    "unique_viewers"
]


def run_prediction(df):

    # ---------------------------------------------------
    # 1. Filter & align columns automatically
    # ---------------------------------------------------
    missing = [c for c in TRAIN_FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    # Keep only the required columns
    df = df[TRAIN_FEATURES].copy()

    # ---------------------------------------------------
    # 2. Transform using saved preprocessor
    # ---------------------------------------------------
    X = preprocessor.transform(df)

    # ---------------------------------------------------
    # 3. Make predictions
    # ---------------------------------------------------
    pred_completion = model_completion.predict(X)
    pred_dropoff = model_dropoff.predict(X)

    # ---------------------------------------------------
    # 4. Return results as dataframe
    # ---------------------------------------------------
    output = df.copy()
    output["pred_avg_completion_rate"] = pred_completion
    output["pred_avg_view_drop_off"] = pred_dropoff

    return output
