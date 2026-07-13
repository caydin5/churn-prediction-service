"""Pydantic schemas for churn prediction requests and responses."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ContractType = Literal["month_to_month", "annual", "two_year"]
PaymentMethod = Literal["credit_card", "bank_transfer", "invoice"]
ProductTier = Literal["starter", "standard", "enterprise"]
PredictionLabel = Literal["retain", "churn"]
RiskBand = Literal["low", "medium", "high"]


class CustomerFeatures(BaseModel):
    """Model input features for one SaaS customer account."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure_months": 8,
                "monthly_charges": 129.0,
                "total_charges": 1032.0,
                "support_tickets_last_90d": 5,
                "contract_type": "month_to_month",
                "payment_method": "credit_card",
                "has_auto_pay": False,
                "product_tier": "standard",
                "seats": 12,
            }
        }
    )

    tenure_months: int = Field(..., ge=0, le=120)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    support_tickets_last_90d: int = Field(..., ge=0, le=100)
    contract_type: ContractType
    payment_method: PaymentMethod
    has_auto_pay: bool
    product_tier: ProductTier
    seats: int = Field(..., ge=1, le=5000)


class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "customer": CustomerFeatures.model_config["json_schema_extra"]["example"],
                "threshold": 0.5,
            }
        }
    )

    customer: CustomerFeatures
    threshold: float = Field(default=0.5, ge=0.0, le=1.0)


class PredictionResponse(BaseModel):
    """Prediction response returned by the model serving endpoint."""

    model_config = ConfigDict(protected_namespaces=())

    prediction: PredictionLabel
    churn_probability: float
    threshold: float
    risk_band: RiskBand
    model_version: str


class ModelInfoResponse(BaseModel):
    """Metadata about the trained model artifact."""

    model_config = ConfigDict(protected_namespaces=())

    model_version: str
    trained_at: str
    feature_columns: list[str]
    metrics: dict[str, float]
