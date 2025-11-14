import joblib
from preprocess import load_data, build_preprocessor, preprocess
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import json

df = load_data("data/fake_kaltura_data.csv")
preprocessor = build_preprocessor()

X, y1, y2 = preprocess(df, preprocessor)

# ------------------------------
# Train Random Forest Model
# ------------------------------
rf = RandomForestRegressor(n_estimators=200, random_state=42)
rf.fit(X, y1)
joblib.dump(rf, "models/rf_model.pkl")

print("RF Completion R2:", r2_score(y1, rf.predict(X)))


# ------------------------------
# Train XGBoost Model
# ------------------------------
xgb_model = xgb.XGBRegressor(n_estimators=200, max_depth=4, learning_rate=0.1)
xgb_model.fit(X, y1)
xgb_model.save_model("models/xgb_model.json")

print("XGB Completion R2:", r2_score(y1, xgb_model.predict(X)))

print("Training complete!")
