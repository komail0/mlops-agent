"""FastAPI inference service for the classifier. Deployed to Kubernetes.

Endpoints:
  GET  /health   liveness: the process is up
  GET  /ready    readiness: 200 only once the model is loaded, 503 otherwise
  GET  /version  application and model version from the environment
  POST /predict  {"features": [30 numbers]} -> class and probability
"""

from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

log = logging.getLogger("inference-api")


class PredictRequest(BaseModel):
    features: list[float] = Field(min_length=1)


class PredictResponse(BaseModel):
    prediction: int
    probability: float | None = None
    model_version: str


def create_app(model_path: str | None = None) -> FastAPI:
    state: dict = {"model": None}

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
        path = Path(model_path or os.environ.get("MODEL_PATH", "models/model.joblib"))
        try:
            state["model"] = joblib.load(path)
            log.info("model loaded from %s", path)
        except Exception as exc:  # the pod stays alive but not ready
            log.error("model not loaded from %s: %s", path, type(exc).__name__)
        yield

    app = FastAPI(title="MLOps inference service", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/ready")
    def ready() -> dict:
        if state["model"] is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        return {"status": "ready"}

    @app.get("/version")
    def version() -> dict:
        return {
            "app_version": os.environ.get("APP_VERSION", "unknown"),
            "model_version": os.environ.get("MODEL_VERSION", "unknown"),
            "model_loaded": state["model"] is not None,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(req: PredictRequest) -> PredictResponse:
        model = state["model"]
        if model is None:
            raise HTTPException(status_code=503, detail="model not loaded")
        expected = int(model.n_features_in_)
        if len(req.features) != expected:
            raise HTTPException(
                status_code=422, detail=f"expected {expected} features, got {len(req.features)}"
            )
        frame = pd.DataFrame([req.features], columns=list(model.feature_names_in_))
        prediction = int(model.predict(frame)[0])
        probability = None
        if hasattr(model, "predict_proba"):
            probability = float(model.predict_proba(frame)[0][prediction])
        return PredictResponse(
            prediction=prediction,
            probability=probability,
            model_version=os.environ.get("MODEL_VERSION", "unknown"),
        )

    return app


app = create_app()
