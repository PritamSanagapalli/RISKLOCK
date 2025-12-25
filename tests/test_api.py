from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_health_endpoint():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict_endpoint_schema():
    payload = {
        "loan_amnt": 10000.0,
        "annual_inc": 50000.0,
        "int_rate": "12.5%",
        "emp_length": "5 years",
        "dti": 15.5,
        "earliest_cr_line": "Jan-2015",
        "term": " 36 months",
        "home_ownership": "RENT",
    }

    resp = client.post("/predict", json=payload)
    assert resp.status_code == 200
    body = resp.json()

    # Validate response fields
    for key in [
        "probability_default",
        "risk_level",
        "top_risk_factors",
        "model_version",
        "data_quality_score",
    ]:
        assert key in body, f"Missing response key: {key}"

    assert isinstance(body["probability_default"], float)
    assert body["risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(body["top_risk_factors"], list)
