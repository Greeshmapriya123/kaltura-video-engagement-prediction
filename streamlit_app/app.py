import streamlit as st
import pandas as pd
import os
import sys
import joblib
import numpy as np

# -------------------------------------------------------------------
# FIX PATH ISSUE → ALLOW IMPORTING FROM src/
# -------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict import run_prediction


# -------------------------------------------------------------------
# STREAMLIT APP UI
# -------------------------------------------------------------------
st.set_page_config(page_title="Kaltura Video Engagement Prediction", layout="wide")

st.title("🎥 Kaltura Video Engagement Prediction App")

st.markdown("""
Upload a CSV file to predict:

- **Average Completion Rate (%)**
- **Engagement Drop-Off Risk**

The model uses:
- Preprocessor: `preprocessor.pkl`
- XGBoost Completion Model: `xgb_completion.pkl`
- XGBoost Dropoff Model: `xgb_dropoff.pkl`
""")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    st.success("CSV uploaded successfully! Processing…")

    # Load data
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Uploaded Data (first 5 rows)")
    st.dataframe(df.head())

    # Run prediction
    try:
        preds = run_prediction(df)

        st.subheader("✅ Predictions")
        st.dataframe(preds)

        # Download option
        csv = preds.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="📥 Download Predictions",
            data=csv,
            file_name="kaltura_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error("❌ Prediction failed. Check your input file format.")
        st.exception(e)

else:
    st.info("Please upload a CSV file to begin.")
