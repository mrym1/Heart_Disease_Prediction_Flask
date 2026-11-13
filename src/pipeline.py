"""
Builds the scikit-learn Pipeline used for every model.

Why a Pipeline?
Preprocessing steps like filling missing values, scaling and one-hot
encoding "learn" something from the data (a median, a mean and standard
deviation, the list of categories). If we fit them on the whole dataset
before cross-validation, the test fold has already influenced them, which
is a small form of data leakage.

Putting them inside a Pipeline means that during cross-validation they are
fitted on the training folds only and then applied to the test fold. It
also gives us one object to save with joblib and load in the Flask app, so
the app does exactly the same preprocessing as in training.
"""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.data import BINARY_FEATURES, CATEGORICAL_FEATURES, NUMERIC_FEATURES


def build_preprocessor():
    """Preprocessing for the three kinds of features."""
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),   # fills the 4 missing `ca` values
        ("scale", StandardScaler()),                    # mean 0, std 1
    ])

    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),  # fills the 2 missing `thal` values
        ("onehot", OneHotEncoder(handle_unknown="ignore")),   # one 0/1 column per category
    ])

    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
        ("bin", "passthrough", BINARY_FEATURES),        # already 0/1, nothing to do
    ])


def build_pipeline(model):
    """Preprocessing followed by the given model."""
    return Pipeline([
        ("prep", build_preprocessor()),
        ("model", model),
    ])


if __name__ == "__main__":
    # Quick check: python -m src.pipeline
    from sklearn.linear_model import LogisticRegression

    from src.data import load_clean

    X, y = load_clean()
    pipe = build_pipeline(LogisticRegression(max_iter=1000))
    pipe.fit(X, y)

    names = pipe.named_steps["prep"].get_feature_names_out()
    print(f"13 input columns become {len(names)} columns after preprocessing:")
    for name in names:
        print("  ", name)

    # The pipeline takes raw values (even with a missing value) and predicts
    one_patient = X.iloc[[49]]          # this patient has ca missing
    print("\nRaw input:\n", one_patient.to_string())
    print("Predicted probability of disease:", round(pipe.predict_proba(one_patient)[0, 1], 3))
