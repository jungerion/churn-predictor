"""Load raw churn data and clean it into a form ready for feature engineering.

Keeping this separate from feature engineering matters: `load_and_clean` fixes
data QUALITY issues (bad types, missing values, whitespace). Feature engineering
(in features/build_features.py) turns clean data into MODEL-READY numeric arrays.
Mixing the two makes both harder to test and reuse.
"""
import pandas as pd
from pathlib import Path

def load_and_clean(raw_path: Path, target_column: str = "Churn") -> pd.DataFrame:
    df = pd.read_csv(raw_path)

    # Telco dataset quirk: TotalCharges is stored as a string with some blank
    # entries for brand-new customers (tenure=0). This is a great example of
    # a "silent" data bug — it looks numeric until you try to cast it.
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        df["TotalCharges"] = df["TotalCharges"].fillna(0)

    # Normalize the target to 0/1 for the model. Checking actual values rather
    # than dtype here — pandas has both "object" and newer "str" dtypes for
    # text columns depending on version, so a dtype check alone is fragile.
    if target_column in df.columns and set(df[target_column].unique()) <= {"Yes", "No"}:
        df[target_column] = df[target_column].map({"Yes": 1, "No": 0})

    # Drop exact duplicate rows, if any.
    df = df.drop_duplicates()

    return df

def train_test_columns(df: pd.DataFrame, target_column: str, id_column: str):
    """Split a cleaned dataframe into feature columns and target, dropping the id."""
    feature_cols = [c for c in df.columns if c not in (target_column, id_column)]
    X = df[feature_cols]
    y = df[target_column]
    return X, y
