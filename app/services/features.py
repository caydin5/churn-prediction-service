"""Shared feature definitions for training and serving."""

from __future__ import annotations

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_charges",
    "support_tickets_last_90d",
    "seats",
]

CATEGORICAL_FEATURES = [
    "contract_type",
    "payment_method",
    "has_auto_pay",
    "product_tier",
]

FEATURE_COLUMNS = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_COLUMN = "churned"
