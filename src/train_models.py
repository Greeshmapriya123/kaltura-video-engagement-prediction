import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

from preprocess import create_preprocessor, preprocess_for_training

# -------------- LOAD YOUR TRAINING DATA ----------------
df = pd.read_csv("data/kaltura_fake_extended.csv")

df = preprocess_for_training(df)

# Columns to predict (example targets)
TARGET = "avg_completion_rate"

# ---------------- CREATE PREPROCESSOR -------------------
preprocessor, num_cols, cat_cols = create_preprocessor(df)
preprocessor.fit(df)

os.makedirs("models", exist_ok=True)
joblib.dump(preprocessor, "models/preprocessor.pkl")
print("Saved: models/preprocessor.pkl")

# ---------------- TRAIN 3 MODELS ------------------------
modes = ["pre", "early", "full"]

for mode in modes:
    model = RandomForestRegressor(n_estimators=300, random_state=42)

    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    model.fit(X, y)

    joblib.dump(model, f"models/model_{mode}.pkl")
    print(f"Saved: models/model_{mode}.pkl")
