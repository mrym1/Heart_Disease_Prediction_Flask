"""
Light hyperparameter tuning with nested cross-validation.

If we tried many settings and reported the best CV score, that score would
be too optimistic: we picked the setting that happened to do best on those
exact folds. Nested CV avoids this:

- OUTER loop (5 folds, the same folds as before): only used to measure
  performance. Each outer test fold is never seen during tuning.
- INNER loop (GridSearchCV, 5 folds inside the outer training data): tries
  every setting in the grid and picks the best one.

So each outer fold gets its own tuned model, tested on patients the tuning
never saw. The outer scores are the honest numbers we report.

We tune for ROC-AUC because it does not depend on the 0.5 cut-off. The
cut-off (and so the recall / precision trade-off) is looked at separately.

Run from the project folder:  python -m src.tune
"""

import warnings
from pathlib import Path

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.compare_models import pretty
from src.data import load_clean
from src.evaluate import CV, SCORING
from src.pipeline import build_pipeline

warnings.filterwarnings("ignore", category=ConvergenceWarning)

RESULTS_PATH = Path(__file__).resolve().parent.parent / "reports" / "cv_results_tuned.csv"

# Inner folds for GridSearchCV (different shuffle from the outer folds)
INNER_CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=1)

# Small grids: a few sensible values per setting, not hundreds of combinations.
# "model__C" means: the parameter C of the pipeline step called "model".
SEARCHES = {
    "Logistic Regression": (
        LogisticRegression(max_iter=1000),
        {"model__C": [0.01, 0.1, 1, 10]},              # smaller C = simpler model
    ),
    "Random Forest": (
        RandomForestClassifier(n_estimators=300, random_state=42),
        {"model__max_depth": [3, 5, None],              # how deep each tree can grow
         "model__min_samples_leaf": [1, 5, 10]},        # min patients in a leaf
    ),
    "SVM (RBF kernel)": (
        SVC(random_state=42),
        {"model__C": [0.1, 1, 10],
         "model__gamma": ["scale", 0.01, 0.1]},         # how "wiggly" the boundary is
    ),
    "XGBoost": (
        XGBClassifier(random_state=42, eval_metric="logloss"),
        {"model__max_depth": [2, 3],
         "model__learning_rate": [0.05, 0.1],
         "model__n_estimators": [100, 300]},
    ),
}


def make_search(model, grid):
    """GridSearchCV wrapped around our pipeline (the INNER loop)."""
    return GridSearchCV(build_pipeline(model), grid, cv=INNER_CV,
                        scoring="roc_auc", n_jobs=-1)


def nested_cv():
    X, y = load_clean()
    rows, chosen = [], {}
    for name, (model, grid) in SEARCHES.items():
        print(f"Tuning {name} ...")
        scores = cross_validate(make_search(model, grid), X, y, cv=CV,
                                scoring=SCORING, return_train_score=True,
                                return_estimator=True)
        row = {"model": name}
        for metric in SCORING:
            row[f"{metric}_mean"] = scores[f"test_{metric}"].mean()
            row[f"{metric}_std"] = scores[f"test_{metric}"].std()
        row["train_accuracy"] = scores["train_accuracy"].mean()
        rows.append(row)
        # Which settings won in each outer fold? (Shows how stable the choice is.)
        chosen[name] = [search.best_params_ for search in scores["estimator"]]
    return pd.DataFrame(rows).set_index("model"), chosen


if __name__ == "__main__":
    results, chosen = nested_cv()
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.round(4).to_csv(RESULTS_PATH)

    pd.set_option("display.width", 200)
    print("\nNested CV (5 outer x 5 inner folds), tuned for ROC-AUC:\n")
    print(pretty(results).to_string())

    print("\nBest settings chosen in each outer fold:")
    for name, params in chosen.items():
        print(f"  {name}:")
        for i, p in enumerate(params, start=1):
            short = {k.replace("model__", ""): v for k, v in p.items()}
            print(f"    fold {i}: {short}")
    print(f"\nSaved to {RESULTS_PATH.name}")
