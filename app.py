"""
Flask app that serves predictions from the saved pipeline.

The pipeline (models/heart_pipeline.joblib) already contains all the
preprocessing (filling missing values, scaling, one-hot encoding), so the
app only has to check the input and pass the 13 raw values to it.

Create the model file first:  python -m src.train_final
Run the app:                  python app.py   (then open http://127.0.0.1:8000)
"""

from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

from src.data import FEATURES

MODEL_PATH = Path(__file__).resolve().parent / "models" / "heart_pipeline.joblib"
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

# Key sent by the web page -> column name the model expects
FIELDS = {
    "Age": "age", "Sex": "sex", "Cp": "cp", "Bp": "trestbps", "Chol": "chol",
    "Fbs": "fbs", "Restecg": "restecg", "Thalach": "thalach", "Exang": "exang",
    "Oldpeak": "oldpeak", "Slope": "slope", "Ca": "ca", "Thal": "thal",
}

# Numeric features: allowed (min, max). These are wide, physically possible
# ranges. The model was trained on age 29-77, blood pressure 94-200,
# cholesterol 126-564, max heart rate 71-202 and oldpeak 0-6.2, so results
# far outside those ranges are less reliable.
NUMERIC_RANGES = {
    "age": (18, 100),
    "trestbps": (80, 220),
    "chol": (100, 600),
    "thalach": (60, 220),
    "oldpeak": (0, 7),
}

# Coded features: allowed codes (same codes as in heart.csv, see the EDA notebook)
ALLOWED_CODES = {
    "sex": {0, 1},             # 0 = female, 1 = male
    "cp": {0, 1, 2, 3},        # 0 = asymptomatic, 1 = atypical, 2 = non-anginal, 3 = typical angina
    "fbs": {0, 1},             # fasting blood sugar > 120 mg/dl
    "restecg": {0, 1, 2},      # 0 = LV hypertrophy, 1 = normal, 2 = ST-T abnormality
    "exang": {0, 1},           # exercise-induced angina
    "slope": {0, 1, 2},        # 0 = downsloping, 1 = flat, 2 = upsloping
    "ca": {0, 1, 2, 3},        # number of major vessels
    "thal": {1, 2, 3},         # 1 = fixed defect, 2 = normal, 3 = reversible defect
}


def validate(data):
    """Check the input. Returns (values, errors): values is a dict of the
    13 features if everything is fine, errors is a list of messages."""
    if not isinstance(data, dict):
        return None, ["Send the input as a JSON object."]

    values, errors = {}, []
    for key, feature in FIELDS.items():
        raw = data.get(key)
        if raw is None or str(raw).strip() == "":
            errors.append(f"{key} is missing.")
            continue
        try:
            number = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{key} must be a number, got {raw!r}.")
            continue

        if feature in NUMERIC_RANGES:
            low, high = NUMERIC_RANGES[feature]
            if not low <= number <= high:
                errors.append(f"{key} must be between {low} and {high}, got {raw}.")
                continue
            values[feature] = number
        else:
            if number not in ALLOWED_CODES[feature]:
                allowed = ", ".join(str(c) for c in sorted(ALLOWED_CODES[feature]))
                errors.append(f"{key} must be one of {allowed}, got {raw}.")
                continue
            values[feature] = int(number)

    return (values if not errors else None), errors


@app.route("/")
def front_page():
    return render_template("index.html")


@app.route("/reply", methods=["POST"])
def reply():
    values, errors = validate(request.get_json(silent=True))
    if errors:
        return jsonify({"errors": errors}), 400

    row = pd.DataFrame([values], columns=FEATURES)   # same column order as in training
    probability = float(model.predict_proba(row)[0, 1])
    prediction = int(probability >= 0.5)              # standard 0.5 cut-off

    return jsonify({
        "prediction": prediction,                     # 1 = heart disease, 0 = no heart disease
        "probability": round(probability, 3),         # model's estimated probability of disease
    })


if __name__ == "__main__":
    app.run(debug=True, port=8000)
