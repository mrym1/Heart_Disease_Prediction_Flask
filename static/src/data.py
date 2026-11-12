"""
Loading and cleaning the heart disease data.

The Kaggle file heart.csv has 1,025 rows, but only 302 of them are unique.
Most patients appear 3 or 4 times. If we split the raw file into train and
test, copies of the same patient end up on both sides, so the model is
partly tested on rows it already saw during training (data leakage).
We therefore remove exact duplicates before doing anything else.
"""

from pathlib import Path

import pandas as pd

# Path to heart.csv, relative to the repo root (works no matter where you run from)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart.csv"

TARGET = "target"

# The 13 input features, in the order the model expects them
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]


def load_raw(path=DATA_PATH):
    """Read the CSV exactly as downloaded."""
    return pd.read_csv(path)


def load_clean(path=DATA_PATH, verbose=False):
    """Read the CSV and drop exact duplicate rows.

    Returns X (features) and y (target) for the unique patients only.
    """
    df = load_raw(path)
    n_before = len(df)

    df = df.drop_duplicates().reset_index(drop=True)

    if verbose:
        print(f"Rows before: {n_before}")
        print(f"Duplicates removed: {n_before - len(df)}")
        print(f"Rows after: {len(df)}")

    X = df[FEATURES]
    y = df[TARGET]
    return X, y


if __name__ == "__main__":
    # Quick check: python src/data.py
    X, y = load_clean(verbose=True)
    counts = y.value_counts().sort_index()
    print("\nClass balance (0 = no disease, 1 = disease):")
    for label, n in counts.items():
        print(f"  {label}: {n} ({n / len(y):.1%})")
