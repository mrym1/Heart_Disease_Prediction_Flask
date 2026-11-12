"""
Loading and cleaning the heart disease data.

The Kaggle file heart.csv is a copy of the UCI Cleveland heart disease data,
but it has three problems that we fix here:

1. Duplicates. The file has 1,025 rows, but only 302 are unique (most
   patients appear 3 or 4 times). If we split the raw file into train and
   test, copies of the same patient land on both sides and the test score
   is inflated (data leakage). We drop exact duplicates first.

2. Reversed target. In this Kaggle version, target = 1 actually means
   "no heart disease". The evidence is in notebooks/01_eda.ipynb. We flip it
   so that 1 = disease, which is what "recall" should be measured on.

3. Missing values hidden as codes. In the original UCI data, 4 values of
   `ca` and 2 values of `thal` were missing ("?"). The Kaggle file stores
   them as ca = 4 and thal = 0, which look like real categories but are not.
   We turn them back into NaN. They are filled in later, inside the model
   pipeline, so the filling is learned from training folds only.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# Path to heart.csv, relative to the repo root (works no matter where you run from)
DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "heart.csv"

# The 13 input features, in the order the model expects them
FEATURES = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal",
]

# Continuous measurements (get scaled)
NUMERIC_FEATURES = ["age", "trestbps", "chol", "thalach", "oldpeak", "ca"]
# Yes/no features (already 0/1)
BINARY_FEATURES = ["sex", "fbs", "exang"]
# Codes for categories with no natural order (get one-hot encoded)
CATEGORICAL_FEATURES = ["cp", "restecg", "slope", "thal"]


def load_raw(path=DATA_PATH):
    """Read the CSV exactly as downloaded."""
    return pd.read_csv(path)


def load_clean(path=DATA_PATH, verbose=False):
    """Read the CSV, drop duplicates, fix the target and missing codes.

    Returns X (13 features) and y (1 = heart disease, 0 = no heart disease).
    """
    df = load_raw(path)
    n_before = len(df)

    # 1. Remove exact duplicate rows
    df = df.drop_duplicates().reset_index(drop=True)

    # 2. Flip the target so that 1 = disease
    y = (1 - df["target"]).rename("disease")

    # 3. Codes that really mean "missing" in the original data
    X = df[FEATURES].copy()
    X["ca"] = X["ca"].replace(4, np.nan)
    X["thal"] = X["thal"].replace(0, np.nan)

    if verbose:
        print(f"Rows before: {n_before}")
        print(f"Duplicates removed: {n_before - len(df)}")
        print(f"Rows after: {len(df)}")
        print(f"Missing values: ca = {X['ca'].isna().sum()}, thal = {X['thal'].isna().sum()}")

    return X, y


if __name__ == "__main__":
    # Quick check: python src/data.py
    X, y = load_clean(verbose=True)
    counts = y.value_counts().sort_index()
    print("\nClass balance (0 = no disease, 1 = disease):")
    for label, n in counts.items():
        print(f"  {label}: {n} ({n / len(y):.1%})")
