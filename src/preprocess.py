import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
import joblib

def load_data(path):
    return pd.read_csv(path)

def build_preprocessor():
    numeric_cols = ["duration_msecs", "count_plays", "count_loads", "load_play_ratio"]
    categorical_cols = ["status", "media_type", "entry_source"]

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )
    return preprocessor

def preprocess(df, preprocessor):
    X = df.drop(["avg_completion_rate", "avg_view_drop_off"], axis=1)
    y1 = df["avg_completion_rate"]
    y2 = df["avg_view_drop_off"]

    X_processed = preprocessor.fit_transform(X)

    joblib.dump(preprocessor, "models/preprocessor.pkl")

    return X_processed, y1, y2

if __name__ == "__main__":
    df = load_data("data/fake_kaltura_data.csv")
    pre = build_preprocessor()
    preprocess(df, pre)
    print("Preprocessing complete!")
