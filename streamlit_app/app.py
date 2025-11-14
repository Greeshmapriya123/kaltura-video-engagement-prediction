import streamlit as st
import pandas as pd
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(ROOT_DIR)

from src.predict import run_prediction

st.set_page_config(page_title="Kaltura Video Engagement Prediction", layout="wide")

st.title("🎥 Kaltura Video Engagement Prediction App")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    
    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    try:
        preds = run_prediction(df)
        st.subheader("Predictions")
        st.dataframe(preds)

        st.download_button(
            "Download Predictions",
            preds.to_csv(index=False).encode("utf-8"),
            "kaltura_predictions.csv",
            "text/csv"
        )

    except Exception as e:
        st.error("Prediction failed")
        st.exception(e)

else:
    st.info("Upload a CSV to begin.")
