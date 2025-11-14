import streamlit as st
import pandas as pd
import joblib
import xgboost as xgb
import matplotlib.pyplot as plt
import numpy as np

st.title("🎥 Kaltura Video Engagement Prediction App")
st.write("Upload a CSV file to predict:")
st.write("- Average Completion Rate")
st.write("- Engagement Drop-Off Risk")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

# Load models
preprocessor = joblib.load("models/preprocessor.pkl")
rf_model = joblib.load("models/rf_model.pkl")

xgb_model = xgb.XGBRegressor()
xgb_model.load_model("models/xgb_model.json")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview")
    st.dataframe(df.head())

    # Preprocess
    X = preprocessor.transform(df)

    # Predictions
    df["pred_rf"] = rf_model.predict(X)
    df["pred_xgb"] = xgb_model.predict(X)

    df["risk"] = np.where(df["pred_rf"] < 30, "HIGH",
                          np.where(df["pred_rf"] < 60, "MEDIUM", "LOW"))

    st.subheader("🚀 Predictions")
    st.dataframe(df)

    # Download button
    csv = df.to_csv(index=False)
    st.download_button("Download Predictions", csv, "predictions.csv")

    # Feature Importance
    st.subheader("📊 Feature Importance (Random Forest)")
    importances = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    idx = np.argsort(importances)[-10:]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(feature_names[idx], importances[idx])
    st.pyplot(fig)
else:
    st.info("Please upload a CSV file.")
