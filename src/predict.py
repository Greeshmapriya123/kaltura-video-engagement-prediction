import pandas as pd
import joblib

def load_artifacts():
    pre = joblib.load("models/preprocessor.pkl")
    model_pre = joblib.load("models/model_pre.pkl")
    model_early = joblib.load("models/model_early.pkl")
    model_full = joblib.load("models/model_full.pkl")
    return pre, model_pre, model_early, model_full


def run_prediction(df, mode):
    pre, model_pre, model_early, model_full = load_artifacts()

    df = df.fillna(0)
    X = pre.transform(df)

    if mode == "PRE":
        model = model_pre
    elif mode == "EARLY":
        model = model_early
    else:
        model = model_full

    predictions = model.predict(X)
    return predictions
