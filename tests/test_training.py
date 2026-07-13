"""Tests for the churn model training pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from app.services.training import load_training_data, train_model


def test_train_model_persists_artifact_and_metrics(tmp_path):
    model_path = tmp_path / "churn_model.joblib"
    metrics_path = tmp_path / "metrics.json"

    artifact = train_model(
        dataset_path=Path("data/customer_churn.csv"),
        model_path=model_path,
        metrics_path=metrics_path,
    )

    assert model_path.exists()
    assert metrics_path.exists()
    assert artifact["model_version"] == "0.1.0"
    assert "roc_auc" in artifact["metrics"]
    assert artifact["metrics"]["f1"] >= 0.5


def test_train_model_creates_output_directories(tmp_path):
    """model_path and metrics_path parent dirs should be created automatically."""

    nested_model = tmp_path / "deep" / "nested" / "model.joblib"
    nested_metrics = tmp_path / "deep" / "nested" / "metrics.json"

    train_model(
        dataset_path=Path("data/customer_churn.csv"),
        model_path=nested_model,
        metrics_path=nested_metrics,
    )

    assert nested_model.exists()
    assert nested_metrics.exists()


def test_load_training_data_rejects_missing_columns(tmp_path):
    """CSV missing required columns should raise ValueError."""

    bad_csv = tmp_path / "bad.csv"
    bad_csv.write_text("col_a,col_b\n1,2\n", encoding="utf-8")

    with pytest.raises(ValueError, match="missing required columns"):
        load_training_data(bad_csv)


def test_load_training_data_rejects_missing_file():
    """A non-existent CSV should raise FileNotFoundError."""

    with pytest.raises(FileNotFoundError):
        load_training_data(Path("/nonexistent/data.csv"))
