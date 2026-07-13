"""Shared test fixtures for the Churn Prediction Service."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on sys.path before importing app modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pytest  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import app, get_model_service, get_optional_model_service  # noqa: E402
from app.schemas import CustomerFeatures  # noqa: E402
from app.services.model_service import ChurnModelService  # noqa: E402
from app.services.training import train_model  # noqa: E402


@pytest.fixture(scope="session")
def trained_model_path(tmp_path_factory) -> Path:
    """Train a model once per test session and return its path."""

    tmp_dir = tmp_path_factory.mktemp("models")
    model_path = tmp_dir / "churn_model.joblib"
    train_model(
        dataset_path=Path("data/customer_churn.csv"),
        model_path=model_path,
        metrics_path=tmp_dir / "metrics.json",
    )
    return model_path


@pytest.fixture(scope="session")
def model_service(trained_model_path: Path) -> ChurnModelService:
    """Load the session-trained model into a ChurnModelService."""

    return ChurnModelService(trained_model_path)


@pytest.fixture()
def api_client(model_service: ChurnModelService):
    """FastAPI test client with both model service dependencies overridden."""

    app.dependency_overrides[get_model_service] = lambda: model_service
    app.dependency_overrides[get_optional_model_service] = lambda: model_service
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture()
def sample_customer() -> CustomerFeatures:
    """A sample high-risk customer for prediction tests."""

    return CustomerFeatures(
        tenure_months=6,
        monthly_charges=139.0,
        total_charges=834.0,
        support_tickets_last_90d=6,
        contract_type="month_to_month",
        payment_method="credit_card",
        has_auto_pay=False,
        product_tier="standard",
        seats=10,
    )
