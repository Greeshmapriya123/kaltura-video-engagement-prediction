import pandas as pd

# Load CSV
def load_data(path):
    df = pd.read_csv(path)
    return df

# Clean + Select features
def preprocess(df):

    # 1️⃣ DROP ALL NON-NUMERIC / TEXT COLUMNS
    drop_cols = [
        "object_id",
        "entry_name",
        "creator_name",
        "month_year",
        "source_file",
        "source_dataset"
    ]

    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 2️⃣ DROP TARGETS DURING TRAINING (they will be added later)
    target_cols = ["avg_completion_rate", "avg_view_drop_off"]
    for col in target_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    # 3️⃣ Fill missing
    df = df.fillna(0)

    # 4️⃣ Keep only numeric columns
    df = df.select_dtypes(include=["float64", "int64"])

    return df
