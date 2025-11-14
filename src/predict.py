import pandas as pd
import joblib

def run_prediction(df):

    # Load artifacts
    scaler = joblib.load("models/preprocessor.pkl")
    model1 = joblib.load("models/xgb_completion.pkl")
    model2 = joblib.load("models/xgb_dropoff.pkl")

    # Columns never used for modeling
    drop_cols = ["object_id", "entry_name", "creator_name"]

    # Remove them if present
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # Remove target columns if uploaded accidentally
    for col in ["avg_completion_rate", "avg_view_drop_off"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    # Preprocess
    X_scaled = scaler.transform(df)

    # Predictions
    df["pred_avg_completion_rate"] = model1.predict(X_scaled)
    df["pred_avg_view_dropoff"] = model2.predict(X_scaled)

    return df
