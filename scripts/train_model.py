"""CLI entry point for training the churn prediction model."""

from __future__ import annotations

from pathlib import Path
import sys
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.services.training import train_model


if __name__ == "__main__":
    artifact = train_model(
        dataset_path=Path("data/customer_churn.csv"),
        model_path=Path("models/churn_model.joblib"),
        metrics_path=Path("models/metrics.json"),
    )
    print("Trained churn model")
    print(f"Model version: {artifact['model_version']}")
    print(f"Metrics: {artifact['metrics']}")
