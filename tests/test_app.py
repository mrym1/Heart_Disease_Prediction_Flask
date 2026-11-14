"""
Tests for the Flask app. Run from the project folder:  python -m pytest
(The model file must exist: python -m src.train_final)
"""

import pytest

from app import app

# A valid patient (row from the dataset, raw Kaggle codes)
VALID = {"Age": 63, "Sex": 1, "Cp": 3, "Bp": 145, "Chol": 233, "Fbs": 1, "Restecg": 0,
         "Thalach": 150, "Exang": 0, "Oldpeak": 2.3, "Slope": 0, "Ca": 0, "Thal": 1}


@pytest.fixture
def client():
    return app.test_client()


def test_valid_input_returns_prediction_and_probability(client):
    response = client.post("/reply", json=VALID)
    assert response.status_code == 200
    data = response.get_json()
    assert data["prediction"] in (0, 1)
    assert 0 <= data["probability"] <= 1
    assert data["prediction"] == int(data["probability"] >= 0.5)


def test_numbers_sent_as_strings_are_accepted(client):
    # The web form sends every value as text
    response = client.post("/reply", json={k: str(v) for k, v in VALID.items()})
    assert response.status_code == 200


def test_missing_field_is_rejected(client):
    data = dict(VALID)
    del data["Chol"]
    response = client.post("/reply", json=data)
    assert response.status_code == 400
    assert "Chol is missing." in response.get_json()["errors"]


def test_out_of_range_and_invalid_codes_are_rejected(client):
    data = dict(VALID, Age=250, Thal=0, Cp="abc")
    response = client.post("/reply", json=data)
    assert response.status_code == 400
    assert len(response.get_json()["errors"]) == 3


def test_non_json_body_is_rejected(client):
    response = client.post("/reply", data="hello")
    assert response.status_code == 400


def test_high_and_low_risk_patients_get_sensible_predictions(client):
    # Many risk signs: blocked vessels, reversible defect, exercise angina...
    high = dict(VALID, Cp=0, Thalach=110, Exang=1, Oldpeak=3.0, Slope=1, Ca=3, Thal=3)
    # Few risk signs
    low = dict(VALID, Sex=0, Cp=2, Thalach=175, Exang=0, Oldpeak=0, Slope=2, Ca=0, Thal=2)
    assert client.post("/reply", json=high).get_json()["probability"] > 0.8
    assert client.post("/reply", json=low).get_json()["probability"] < 0.2
