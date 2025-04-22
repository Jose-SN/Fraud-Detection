# test_app.py

import pytest
from fastapi.testclient import TestClient
from app import app, features
import joblib

client = TestClient(app)

# Sample input using only selected 9 features
valid_transaction = {
    "TransactionAmt": 50.0,
    "card1": 1234,
    "card2": 567.0,
    "dist1": 10.0,
    "C1": 1,
    "C2": 2,
    "D1": 5.0,
    "V1": 0.45,
    "V2": 0.65
}

# 1. Test health check
def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "OK"

# 2. Test root endpoint
def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()

# 3. Test prediction endpoint with valid input
def test_valid_prediction():
    response = client.post("/predict", json=valid_transaction)
    assert response.status_code == 200
    json_data = response.json()
    assert "is_fraud" in json_data
    assert "fraud_probability" in json_data
    assert 0.0 <= json_data["fraud_probability"] <= 1.0

# 4. Test prediction endpoint with missing fields
def test_invalid_prediction_missing_fields():
    bad_data = valid_transaction.copy()
    del bad_data["TransactionAmt"]
    response = client.post("/predict", json=bad_data)
    assert response.status_code == 422  # Unprocessable Entity due to schema validation

# 5. Test model and scaler loading
def test_model_scaler_loading():
    model = joblib.load("best_fraud_detection_model.pkl")
    scaler = joblib.load("scaler.pkl")
    assert model is not None
    assert scaler is not None

# 6. Test if feature order matches API
def test_features_alignment():
    assert isinstance(features, list)
    assert len(features) == 9
    for f in valid_transaction:
        assert f in features
