"""
Stratified k-fold cross-validation.

With only 302 patients, one 80/20 split gives a test set of about 60
patients, and the score depends a lot on which patients happen to land in
it. Cross-validation uses every patient for testing exactly once:

- the data is split into 5 parts ("folds"),
- the model is trained on 4 folds and tested on the 5th,
- this repeats 5 times, so each fold is the test set once,
- we report the mean and standard deviation of the 5 test scores.

"Stratified" means every fold keeps the same share of disease / no disease
patients as the full data (about 46% / 54%).
"""

import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate

# Same folds for every model, so the comparison is fair
CV = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Metrics we report. "1 = disease" is the positive class.
SCORING = ["accuracy", "precision", "recall", "f1", "roc_auc"]


def cv_scores(pipeline, X, y, cv=CV):
    """Run cross-validation and return a table: one row per fold."""
    results = cross_validate(pipeline, X, y, cv=cv, scoring=SCORING,
                             return_train_score=True)
    return pd.DataFrame(results)


def summarise(scores):
    """Turn per-fold scores into 'mean ± std' for each test metric."""
    return {
        metric: f"{scores[f'test_{metric}'].mean():.3f} ± {scores[f'test_{metric}'].std():.3f}"
        for metric in SCORING
    }


if __name__ == "__main__":
    # Quick check: python -m src.evaluate
    # Re-test the ORIGINAL model (SVC with default settings), but honestly:
    # no duplicates, preprocessing inside the pipeline, 5-fold CV.
    from sklearn.svm import SVC

    from src.data import load_clean
    from src.pipeline import build_pipeline

    X, y = load_clean()
    scores = cv_scores(build_pipeline(SVC()), X, y)

    print("Accuracy in each fold:")
    for i, (train, test) in enumerate(zip(scores["train_accuracy"], scores["test_accuracy"]), start=1):
        print(f"  fold {i}: train {train:.3f}   test {test:.3f}")

    print(f"\nMean train accuracy: {scores['train_accuracy'].mean():.3f}")
    print(f"Mean test accuracy:  {scores['test_accuracy'].mean():.3f} "
          f"± {scores['test_accuracy'].std():.3f}")
