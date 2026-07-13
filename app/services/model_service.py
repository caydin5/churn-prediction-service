"""Model loading and prediction service."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from app.schemas import CustomerFeatures, PredictionResponse, RiskBand


class ModelNotFoundError(RuntimeError):
    """Raised when the trained model artifact is missing."""


class ChurnModelService:
    """Loads a persisted model artifact and serves churn predictions."""

    def __init__(self, model_path: Path) -> None:
        if not model_path.exists():
            raise ModelNotFoundError(
                f"Model artifact not found at {model_path}. Run `make train` first."
            )

        artifact = joblib.load(model_path)
        self.pipeline = artifact["pipeline"]
        self.model_version = artifact["model_version"]
        self.trained_at = artifact["trained_at"]
        self.feature_columns = artifact["feature_columns"]
        self.metrics = artifact["metrics"]

    def predict(self, customer: CustomerFeatures, *, threshold: float) -> PredictionResponse:
        """Predict churn probability for a single customer.

        When the predicted probability equals the threshold exactly,
        the customer is classified as "churn" (>= comparison).
        """

        frame = pd.DataFrame([customer.model_dump()])
        probability = float(self.pipeline.predict_proba(frame)[0][1])
        prediction = "churn" if probability >= threshold else "retain"

        return PredictionResponse(
            prediction=prediction,
            churn_probability=round(probability, 4),
            threshold=threshold,
            risk_band=risk_band(probability),
            model_version=self.model_version,
        )

    def info(self) -> dict[str, Any]:
        """Return model metadata."""

        return {
            "model_version": self.model_version,
            "trained_at": self.trained_at,
            "feature_columns": self.feature_columns,
            "metrics": self.metrics,
        }

    def feature_importance(self) -> list[dict[str, Any]]:
        """Extract feature importances from the logistic regression coefficients.

        Returns a list of dicts sorted by absolute coefficient magnitude
        (most impactful feature first). Each dict contains:
        - feature: the feature name (including one-hot encoded categories)
        - coefficient: the raw logistic regression coefficient
        - direction: "increases_churn" or "decreases_churn"
        """

        preprocessor = self.pipeline.named_steps["preprocessor"]
        classifier = self.pipeline.named_steps["classifier"]

        # Get feature names from the ColumnTransformer.
        feature_names = preprocessor.get_feature_names_out().tolist()
        coefficients = classifier.coef_[0].tolist()

        importance = []
        for name, coef in zip(feature_names, coefficients):
            # Strip the transformer prefix (e.g., "numeric__" or "categorical__").
            clean_name = name.split("__", 1)[-1] if "__" in name else name

            importance.append({
                "feature": clean_name,
                "coefficient": round(coef, 4),
                "direction": "increases_churn" if coef > 0 else "decreases_churn",
            })

        importance.sort(key=lambda x: abs(x["coefficient"]), reverse=True)
        return importance


def risk_band(probability: float) -> RiskBand:
    """Map a churn probability to a simple business-facing risk band.

    These thresholds (0.4 and 0.7) are heuristic, not calibrated.
    In production, calibrate against historical churn rates.
    """

    if probability >= 0.7:
        return "high"
    if probability >= 0.4:
        return "medium"
    return "low"
