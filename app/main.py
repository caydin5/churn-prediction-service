"""FastAPI entry point for the Churn Prediction Service."""

from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Request
from fastapi.responses import JSONResponse

from app.schemas import ModelInfoResponse, PredictionRequest, PredictionResponse
from app.services.model_service import ChurnModelService, ModelNotFoundError

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Churn Prediction Service",
    description="End-to-end ML training and FastAPI model serving for SaaS churn prediction.",
    version="0.1.0",
)


@app.exception_handler(ModelNotFoundError)
async def model_not_found_handler(request: Request, exc: ModelNotFoundError) -> JSONResponse:
    """Return a clean 503 when the model artifact is missing."""

    logger.warning(
        "Model not found on %s %s: %s",
        request.method,
        request.url.path,
        exc,
    )
    return JSONResponse(
        status_code=503,
        content={"status": "error", "message": str(exc)},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Return clean JSON for unexpected errors."""

    logger.error(
        "Unhandled exception on %s %s: %s",
        request.method,
        request.url.path,
        exc,
        exc_info=True,
    )
    return JSONResponse(
        status_code=500,
        content={"status": "error", "message": "An internal error occurred."},
    )


@lru_cache
def load_model_service() -> ChurnModelService:
    """Load and cache the trained model service from disk."""

    model_path = Path(os.getenv("MODEL_PATH", "models/churn_model.joblib"))
    return ChurnModelService(model_path)


def get_model_service() -> ChurnModelService:
    """Return the model service; raises ModelNotFoundError on missing artifact."""

    return load_model_service()


def get_optional_model_service() -> Optional[ChurnModelService]:
    """Return the model service if available; otherwise return None."""

    try:
        return load_model_service()
    except ModelNotFoundError:
        return None


@app.get("/health")
def health_check(
    service: Optional[ChurnModelService] = Depends(get_optional_model_service),
) -> dict[str, object]:
    """Health check endpoint."""

    if service is None:
        return {"status": "degraded", "model_loaded": False}

    return {
        "status": "ok",
        "model_loaded": True,
        "model_version": service.model_version,
    }


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info(
    service: ChurnModelService = Depends(get_model_service),
) -> ModelInfoResponse:
    """Return metadata and evaluation metrics for the trained model."""

    return ModelInfoResponse(**service.info())


@app.post("/predict", response_model=PredictionResponse)
def predict(
    payload: PredictionRequest,
    service: ChurnModelService = Depends(get_model_service),
) -> PredictionResponse:
    """Predict whether one customer is likely to churn."""

    return service.predict(payload.customer, threshold=payload.threshold)


@app.post("/reload-model")
def reload_model() -> dict[str, str]:
    """Clear the cached model and reload from disk.

    Use this after retraining to pick up the new artifact
    without restarting the server.
    """

    load_model_service.cache_clear()
    _ = load_model_service()  # eagerly reload to fail fast
    logger.info("Model reloaded successfully.")
    return {"status": "ok", "message": "Model reloaded."}


@app.get("/feature-importance")
def feature_importance(
    service: ChurnModelService = Depends(get_model_service),
) -> list[dict]:
    """Return logistic regression coefficients sorted by absolute magnitude.

    Each entry includes the feature name, coefficient value, and whether
    the feature increases or decreases churn probability.
    """

    return service.feature_importance()
