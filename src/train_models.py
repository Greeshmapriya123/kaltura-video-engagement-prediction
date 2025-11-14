import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor
import os

print("=== TRAINING STARTED ===")

# ==========================
# Load Training Data
# ==========================
DATA_PATH = "data/kaltura_fake_extended.csv"

print(f"Loading data from: {DATA_PATH}")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("Data loaded:", df.shape)

# ==========================
# Define Columns
# ==========================

TARGET_COMPLETION = "avg_completion_rate"
TARGET_DROPOFF = "avg_view_drop_off"

DROP_COLS = ["object_id", "entry_name", "creator_name"]

print("Preparing feature matrix...")

X = df.drop(columns=DROP_COLS + [TARGET_COMPLETION, TARGET_DROPOFF])
y1 = df[TARGET_COMPLETION]
y2 = df[TARGET_DROPOFF]

print("Features:", X.shape)
print("Target 1:", y1.shape)
print("Target 2:", y2.shape)

# ==========================
# Preprocessing
# ==========================
print("Fitting scaler...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

os.makedirs("models", exist_ok=True)

print("Saving scaler → models/preprocessor.pkl")
joblib.dump(scaler, "models/preprocessor.pkl")

# ==========================
# Model 1 — Completion
# ==========================
print("Training XGBoost completion model...")

model1 = XGBRegressor(
    n_estimators=400,
    learning_rate=0.07,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model1.fit(X_scaled, y1)
print("Saving xgb_completion.pkl")
joblib.dump(model1, "models/xgb_completion.pkl")

# ==========================
# Model 2 — Dropoff
# ==========================
print("Training XGBoost dropoff model...")

model2 = XGBRegressor(
    n_estimators=400,
    learning_rate=0.07,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.8,
    random_state=42,
    objective="reg:squarederror"
)

model2.fit(X_scaled, y2)
print("Saving xgb_dropoff.pkl")
joblib.dump(model2, "models/xgb_dropoff.pkl")

print("=== TRAINING FINISHED SUCCESSFULLY ===")
