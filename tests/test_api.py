"""API tests for the Churn Prediction Service."""

from __future__ import annotations

from fastapi.testclient import TestClient

from app.main import app, get_optional_model_service


def test_health_endpoint_with_loaded_model(api_client: TestClient):
    response = api_client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["model_loaded"] is True


def test_health_endpoint_degraded_without_model():
    """When no model is loaded, /health should return status=degraded."""

    app.dependency_overrides[get_optional_model_service] = lambda: None
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "degraded"
    assert response.json()["model_loaded"] is False
    app.dependency_overrides.clear()


def test_model_info_endpoint_returns_metrics(api_client: TestClient):
    response = api_client.get("/model-info")

    data = response.json()
    assert response.status_code == 200
    assert data["model_version"] == "0.1.0"
    assert "f1" in data["metrics"]
    assert "feature_columns" in data


def test_predict_endpoint_returns_churn_probability(api_client: TestClient):
    response = api_client.post(
        "/predict",
        json={
            "customer": {
                "tenure_months": 8,
                "monthly_charges": 129.0,
                "total_charges": 1032.0,
                "support_tickets_last_90d": 5,
                "contract_type": "month_to_month",
                "payment_method": "credit_card",
                "has_auto_pay": False,
                "product_tier": "standard",
                "seats": 12,
            },
            "threshold": 0.5,
        },
    )

    data = response.json()
    assert response.status_code == 200
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["prediction"] in {"retain", "churn"}
    assert data["risk_band"] in {"low", "medium", "high"}
    assert data["model_version"] == "0.1.0"


def test_predict_endpoint_rejects_invalid_contract_type(api_client: TestClient):
    response = api_client.post(
        "/predict",
        json={
            "customer": {
                "tenure_months": 8,
                "monthly_charges": 129.0,
                "total_charges": 1032.0,
                "support_tickets_last_90d": 5,
                "contract_type": "weekly",
                "payment_method": "credit_card",
                "has_auto_pay": False,
                "product_tier": "standard",
                "seats": 12,
            }
        },
    )

    assert response.status_code == 422


def test_predict_endpoint_rejects_negative_tenure(api_client: TestClient):
    """tenure_months has ge=0 constraint."""

    response = api_client.post(
        "/predict",
        json={
            "customer": {
                "tenure_months": -1,
                "monthly_charges": 129.0,
                "total_charges": 1032.0,
                "support_tickets_last_90d": 5,
                "contract_type": "month_to_month",
                "payment_method": "credit_card",
                "has_auto_pay": False,
                "product_tier": "standard",
                "seats": 12,
            }
        },
    )

    assert response.status_code == 422


def test_predict_endpoint_rejects_empty_body(api_client: TestClient):
    response = api_client.post("/predict", json={})

    assert response.status_code == 422


def test_predict_endpoint_rejects_invalid_threshold(api_client: TestClient):
    """Threshold must be between 0.0 and 1.0."""

    response = api_client.post(
        "/predict",
        json={
            "customer": {
                "tenure_months": 8,
                "monthly_charges": 129.0,
                "total_charges": 1032.0,
                "support_tickets_last_90d": 5,
                "contract_type": "month_to_month",
                "payment_method": "credit_card",
                "has_auto_pay": False,
                "product_tier": "standard",
                "seats": 12,
            },
            "threshold": 1.5,
        },
    )

    assert response.status_code == 422


def test_model_info_503_without_model():
    """When no model is loaded, /model-info should return 503."""

    from app.main import get_model_service
    from app.services.model_service import ModelNotFoundError

    def _raise():
        raise ModelNotFoundError("No model available.")

    app.dependency_overrides[get_model_service] = _raise
    client = TestClient(app, raise_server_exceptions=False)

    response = client.get("/model-info")

    assert response.status_code == 503
    assert "No model available" in response.json()["message"]
    app.dependency_overrides.clear()


def test_reload_model_endpoint(api_client: TestClient, trained_model_path):
    """POST /reload-model should clear the cache and return success."""

    import os
    os.environ["MODEL_PATH"] = str(trained_model_path)

    # Clear any dependency overrides so the real load path runs
    from app.main import load_model_service
    load_model_service.cache_clear()
    app.dependency_overrides.clear()

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post("/reload-model")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    load_model_service.cache_clear()


def test_feature_importance_endpoint(api_client: TestClient):
    """GET /feature-importance should return sorted coefficients."""

    response = api_client.get("/feature-importance")

    data = response.json()
    assert response.status_code == 200
    assert len(data) > 0

    # Each entry should have feature, coefficient, and direction
    first = data[0]
    assert "feature" in first
    assert "coefficient" in first
    assert first["direction"] in {"increases_churn", "decreases_churn"}

    # Should be sorted by absolute coefficient (descending)
    abs_coefs = [abs(entry["coefficient"]) for entry in data]
    assert abs_coefs == sorted(abs_coefs, reverse=True)
