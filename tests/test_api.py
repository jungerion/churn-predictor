from fastapi.testclient import TestClient
import churn_predictor.api.main as api_module

client = TestClient(api_module.app)

SAMPLE_CUSTOMER = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 70.35,
    "TotalCharges": 845.5,
}


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_predict_returns_expected_shape(monkeypatch):
    monkeypatch.setattr(
        api_module,
        "predict_one",
        lambda features: {"churn_prediction": True, "churn_probability": 0.87},
    )
    response = client.post("/predict", json=SAMPLE_CUSTOMER)
    assert response.status_code == 200
    body = response.json()
    assert "churn_prediction" in body
    assert "churn_probability" in body
    assert 0 <= body["churn_probability"] <= 1


def test_predict_rejects_malformed_input():
    bad_payload = {"gender": "Female"}  # missing required fields
    response = client.post("/predict", json=bad_payload)
    assert response.status_code == 422