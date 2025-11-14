import pandas as pd
import joblib
import xgboost as xgb

def predict_new(csv_path):
    df = pd.read_csv(csv_path)

    preprocessor = joblib.load("models/preprocessor.pkl")
    X = preprocessor.transform(df)

    rf = joblib.load("models/rf_model.pkl")

    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("models/xgb_model.json")

    df["pred_completion_rf"] = rf.predict(X)
    df["pred_completion_xgb"] = xgb_model.predict(X)

    return df

if __name__ == "__main__":
    preds = predict_new("data/fake_kaltura_data.csv")
    print(preds.head())
