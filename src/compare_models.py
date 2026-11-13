"""
Compare four models with the same preprocessing and the same 5 CV folds.

All models use default settings here. Tuning comes later (GridSearchCV).
A "dummy" model that always predicts the most common class is included as
a baseline: any real model must beat it to be useful.

Run from the project folder:  python -m src.compare_models
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data import load_clean
from src.evaluate import SCORING, cv_scores
from src.pipeline import build_pipeline

# The baseline never predicts "disease", so its precision is 0/0. sklearn
# warns about this and reports 0. That is expected, so we hide the warning.
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "reports" / "cv_results_default.csv"

MODELS = {
    "Baseline (always 'no disease')": DummyClassifier(strategy="most_frequent"),
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Random Forest": RandomForestClassifier(random_state=42),
    "SVM (RBF kernel)": SVC(random_state=42),
    "XGBoost": XGBClassifier(random_state=42, eval_metric="logloss"),
}


def compare(models=MODELS):
    X, y = load_clean()
    rows = []
    for name, model in models.items():
        scores = cv_scores(build_pipeline(model), X, y)
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        row["train_accuracy"] = scores["train_accuracy"].mean()
        rows.append(row)
    return pd.DataFrame(rows).set_index("model")


def pretty(results):
    """'mean ± std' strings, easier to read and to copy into the README."""
    table = pd.DataFrame(index=results.index)
    for metric in SCORING:
        table[metric] = [f"{m:.3f} ± {s:.3f}" for m, s in
                         zip(results[f"{metric}_mean"], results[f"{metric}_std"])]
    table["train acc."] = results["train_accuracy"].map("{:.3f}".format)
    return table


if __name__ == "__main__":
    results = compare()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH)

    pd.set_option("display.width", 200)
    print("Stratified 5-fold CV, 302 patients, default settings:\n")
    print(pretty(results).to_string())
    print(f"\nSaved to {RESULTS_PATH.name}")
