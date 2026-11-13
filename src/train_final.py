"""
Train the final model on all 302 patients and save it.

Cross-validation (src/tune.py) told us HOW WELL this approach works on
unseen patients. For the model we actually use in the app, we now train on
all the data, so it learns from every patient. Its expected performance is
the cross-validation result, not a new number.

The value of C is chosen again with GridSearchCV on all the data (same small
grid as in tune.py).

Run from the project folder:  python -m src.train_final
"""

from pathlib import Path

import joblib

from src.data import load_clean
from src.tune import SEARCHES, make_search

MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "heart_pipeline.joblib"
FINAL_MODEL = "Logistic Regression"


def train_final():
    X, y = load_clean()
    model, grid = SEARCHES[FINAL_MODEL]
    search = make_search(model, grid)
    search.fit(X, y)
    return search.best_estimator_, search.best_params_


if __name__ == "__main__":
    pipeline, params = train_final()
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    print(f"Final model: {FINAL_MODEL}, best settings: {params}")
    print(f"Saved to {MODEL_PATH.relative_to(MODEL_PATH.parent.parent)}")
