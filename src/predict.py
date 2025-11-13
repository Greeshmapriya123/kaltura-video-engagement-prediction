import joblib
import pandas as pd
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from preprocess import load_fake_data, build_preprocessor, preprocess_data

df = load_fake_data("data/fake_kaltura_data.csv")
preprocessor = build_preprocessor()
X, y1, y2 = preprocess_data(df, preprocessor)

# -------------------------
# Train Random Forest Model
# -------------------------
rf = RandomForestRegressor(n_estimators=150, random_state=42)
rf.fit(X, y1)
joblib.dump(rf, "models/rf_model.pkl")

# Evaluate
y_pred = rf.predict(X)
print("📌 RF Completion R2:", r2_score(y1, y_pred))

# -------------------------
# Train DNN Model
# -------------------------
dnn = Sequential([
    Dense(64, activation="relu", input_shape=(X.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])

dnn.compile(optimizer="adam", loss="mse")
dnn.fit(X, y1, epochs=30, verbose=0)
dnn.save("models/dnn_model.keras")

print("Training complete!")
