"""Training pipeline for the churn prediction model."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.services.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    TARGET_COLUMN,
)


MODEL_VERSION = "0.1.0"


def load_training_data(dataset_path: Path) -> pd.DataFrame:
    """Load and validate the churn training dataset."""

    data = pd.read_csv(dataset_path)
    required_columns = set(FEATURE_COLUMNS + [TARGET_COLUMN])
    missing = sorted(required_columns - set(data.columns))
    if missing:
        raise ValueError(f"Training data is missing required columns: {missing}")
    return data


def build_pipeline() -> Pipeline:
    """Build a preprocessing + logistic regression pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_FEATURES),
            ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    classifier = LogisticRegression(
        solver="liblinear",
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def rounded_metric(value: float) -> float:
    """Return a JSON-friendly rounded metric value."""

    return float(round(float(value), 4))


def train_model(
    *,
    dataset_path: Path,
    model_path: Path,
    metrics_path: Path,
) -> dict[str, Any]:
    """Train, evaluate, and persist a churn model artifact."""

    data = load_training_data(dataset_path)
    x = data[FEATURE_COLUMNS]
    y = data[TARGET_COLUMN]

    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        random_state=42,
        stratify=y,
    )

    pipeline = build_pipeline()
    pipeline.fit(x_train, y_train)

    probabilities = pipeline.predict_proba(x_test)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    metrics = {
        "accuracy": rounded_metric(accuracy_score(y_test, predictions)),
        "precision": rounded_metric(precision_score(y_test, predictions, zero_division=0)),
        "recall": rounded_metric(recall_score(y_test, predictions, zero_division=0)),
        "f1": rounded_metric(f1_score(y_test, predictions, zero_division=0)),
        "roc_auc": rounded_metric(roc_auc_score(y_test, probabilities)),
    }

    trained_at = datetime.now(timezone.utc).isoformat()
    artifact = {
        "pipeline": pipeline,
        "model_version": MODEL_VERSION,
        "trained_at": trained_at,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(artifact, model_path)

    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return artifact
