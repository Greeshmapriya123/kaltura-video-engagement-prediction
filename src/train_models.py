from preprocess import load_data, preprocess
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import joblib
import pandas as pd

# Load CSV
df = load_data("data/kaltura_fake_extended.csv")

# Save targets separately
y1 = df["avg_completion_rate"]
y2 = df["avg_view_drop_off"]

# Preprocess input features
X = preprocess(df)

# Scale inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save preprocessor
joblib.dump(scaler, "models/preprocessor.pkl")
print("Saved preprocessor")

# Train models
rf1 = RandomForestRegressor(n_estimators=300, random_state=42)
rf1.fit(X_scaled, y1)
joblib.dump(rf1, "models/xgb_completion.pkl")
print("Saved xgb_completion.pkl")

rf2 = RandomForestRegressor(n_estimators=300, random_state=42)
rf2.fit(X_scaled, y2)
joblib.dump(rf2, "models/xgb_dropoff.pkl")
print("Saved xgb_dropoff.pkl")
