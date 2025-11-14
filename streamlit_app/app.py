import streamlit as st
import pandas as pd
from src.predict import run_prediction

st.set_page_config(page_title="Kaltura Video Engagement Predictor", layout="wide")

st.title("🎬 Kaltura Video Engagement Prediction App")

st.write("""
Upload a CSV file containing Kaltura video metadata or engagement features.
Select the mode to generate predictions.
""")

mode = st.selectbox("Select Prediction Mode:", ["PRE", "EARLY", "FULL"])

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview:", df.head())

    if st.button("Generate Predictions"):
        preds = run_prediction(df, mode)
        df[f"predicted_{mode.lower()}"] = preds

        st.success("Prediction Completed!")
        st.write(df.head())

        csv_download = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions", csv_download, "predictions.csv")
