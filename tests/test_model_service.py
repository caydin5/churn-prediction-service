"""Tests for model loading and prediction behavior."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.schemas import CustomerFeatures
from app.services.model_service import ChurnModelService, ModelNotFoundError, risk_band


# --- risk_band ---


def test_risk_band_low():
    assert risk_band(0.0) == "low"
    assert risk_band(0.2) == "low"
    assert risk_band(0.39) == "low"


def test_risk_band_medium_boundary():
    """0.4 is the transition point from low to medium."""

    assert risk_band(0.4) == "medium"
    assert risk_band(0.5) == "medium"
    assert risk_band(0.69) == "medium"


def test_risk_band_high_boundary():
    """0.7 is the transition point from medium to high."""

    assert risk_band(0.7) == "high"
    assert risk_band(0.8) == "high"
    assert risk_band(1.0) == "high"


# --- ModelNotFoundError ---


def test_model_not_found_on_missing_path():
    with pytest.raises(ModelNotFoundError, match="Model artifact not found"):
        ChurnModelService(Path("/nonexistent/model.joblib"))


# --- predict ---


def test_predict_returns_probability_and_label(
    model_service: ChurnModelService,
    sample_customer: CustomerFeatures,
):
    response = model_service.predict(sample_customer, threshold=0.5)

    assert 0.0 <= response.churn_probability <= 1.0
    assert response.prediction in {"retain", "churn"}
    assert response.model_version == "0.1.0"


def test_predict_threshold_zero_always_churns(
    model_service: ChurnModelService,
    sample_customer: CustomerFeatures,
):
    """With threshold=0.0, any non-zero probability should predict churn."""

    response = model_service.predict(sample_customer, threshold=0.0)

    assert response.prediction == "churn"


def test_predict_threshold_one_always_retains(
    model_service: ChurnModelService,
    sample_customer: CustomerFeatures,
):
    """With threshold=1.0, only a probability of exactly 1.0 would churn.
    Any normal model probability < 1.0 should predict retain.
    """

    response = model_service.predict(sample_customer, threshold=1.0)

    # A logistic regression should never produce exactly 1.0
    assert response.prediction == "retain"


# --- info ---


def test_info_returns_metadata(model_service: ChurnModelService):
    info = model_service.info()

    assert info["model_version"] == "0.1.0"
    assert "trained_at" in info
    assert "feature_columns" in info
    assert "roc_auc" in info["metrics"]


# --- feature_importance ---


def test_feature_importance_returns_all_features(model_service: ChurnModelService):
    """Should return one entry per feature (including one-hot encoded categories)."""

    importance = model_service.feature_importance()

    assert len(importance) > 0
    features = [entry["feature"] for entry in importance]

    # At minimum, the numeric features should be present
    assert "tenure_months" in features
    assert "monthly_charges" in features

    # Each entry should have the required keys
    for entry in importance:
        assert "feature" in entry
        assert "coefficient" in entry
        assert entry["direction"] in {"increases_churn", "decreases_churn"}

    # Sorted by absolute coefficient descending
    abs_coefs = [abs(e["coefficient"]) for e in importance]
    assert abs_coefs == sorted(abs_coefs, reverse=True)


# --- model behavior ---


def test_high_risk_customer_scores_higher_than_low_risk(
    model_service: ChurnModelService,
):
    """A clearly high-risk customer should score higher than a low-risk one."""

    from app.schemas import CustomerFeatures

    high_risk = CustomerFeatures(
        tenure_months=2,
        monthly_charges=150.0,
        total_charges=300.0,
        support_tickets_last_90d=8,
        contract_type="month_to_month",
        payment_method="credit_card",
        has_auto_pay=False,
        product_tier="starter",
        seats=5,
    )

    low_risk = CustomerFeatures(
        tenure_months=48,
        monthly_charges=50.0,
        total_charges=2400.0,
        support_tickets_last_90d=0,
        contract_type="two_year",
        payment_method="bank_transfer",
        has_auto_pay=True,
        product_tier="enterprise",
        seats=50,
    )

    high_result = model_service.predict(high_risk, threshold=0.5)
    low_result = model_service.predict(low_risk, threshold=0.5)

    assert high_result.churn_probability > low_result.churn_probability
