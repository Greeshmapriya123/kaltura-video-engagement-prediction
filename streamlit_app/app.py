import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import xgboost as xgb
import matplotlib.pyplot as plt

# --------------------------------------------------
# Load models and preprocessor
# --------------------------------------------------

rf_model = joblib.load("model/rf_model.pkl")
preprocessor = joblib.load("preprocessor/preprocessor.pkl")

try:
    dnn_model = tf.keras.models.load_model("model/dnn_model.keras")
except:
    dnn_model = None

try:
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model("model/xgb_model.json")
except:
    xgb_model = None

# --------------------------------------------------
# Streamlit UI
# --------------------------------------------------

st.title("🎥 Kaltura Video Engagement Prediction App")
st.write("Upload a Kaltura CSV to predict:")
st.write("- **Average Completion Rate**")
st.write("- **View Drop-Off**")
st.write("- **Engagement Risk Levels**")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.subheader("📄 Preview of Uploaded Data")
    st.dataframe(data.head())

    # Preprocess
    X = preprocessor.transform(data)

    # Predictions
    st.subheader("🚀 Predictions")

    predictions = {}

    predictions["Random Forest"] = rf_model.predict(X)

    if dnn_model:
        predictions["DNN Model"] = dnn_model.predict(X).flatten()

    if xgb_model:
        predictions["XGBoost"] = xgb_model.predict(X)

    pred_df = pd.DataFrame(predictions)
    st.dataframe(pred_df)

    # Risk labeling example
    st.subheader("⚠️ Risk Assessment")
    pred_df["RiskLevel"] = np.where(pred_df["Random Forest"] < 30, "HIGH",
                             np.where(pred_df["Random Forest"] < 60, "MEDIUM", "LOW"))
    st.dataframe(pred_df)

    # Download predictions
    st.download_button(
        "Download Predictions",
        pred_df.to_csv(index=False),
        "kaltura_predictions.csv",
        "text/csv"
    )

    # Feature Importance (RF)
    st.subheader("📊 Feature Importance (Random Forest)")
    importances = rf_model.feature_importances_
    feature_names = preprocessor.get_feature_names_out()

    fig, ax = plt.subplots(figsize=(8, 6))
    idx = np.argsort(importances)[-15:]
    ax.barh(feature_names[idx], importances[idx])
    ax.set_title("Top 15 Important Features")
    st.pyplot(fig)

else:
    st.info("Please upload a CSV file to begin.")
