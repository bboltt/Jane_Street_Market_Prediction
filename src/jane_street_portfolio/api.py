"""FastAPI application for deployment-grade model serving."""

from __future__ import annotations

from typing import Any

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .service import JaneStreetModelService


class PredictRequest(BaseModel):
    """Input payload containing feature rows for batch inference."""

    rows: list[dict[str, float]] = Field(..., min_length=1)


class PredictResponse(BaseModel):
    """Prediction payload with probabilities and recommended actions."""

    probabilities: list[float]
    actions: list[int]


def create_app(service: JaneStreetModelService | None = None) -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(title="Jane Street Market Prediction Service", version="1.0.0")
    model_service = service or JaneStreetModelService()

    @app.get("/health")
    def health() -> dict[str, str]:
        """Liveness endpoint used by orchestrators and load balancers."""
        return {"status": "ok"}

    @app.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest) -> PredictResponse:
        """Generate model predictions for a list of feature dictionaries."""
        try:
            frame = pd.DataFrame(payload.rows)
            result = model_service.predict(frame)
        except Exception as exc:  # pragma: no cover - surfaced as API error.
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        return PredictResponse(
            probabilities=[float(value) for value in result.probabilities],
            actions=[int(value) for value in result.actions],
        )

    @app.post("/load")
    def load_model() -> dict[str, Any]:
        """Load the latest serialized model artifact into memory."""
        try:
            model_service.load()
        except Exception as exc:  # pragma: no cover - surfaced as API error.
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        return {"status": "loaded"}

    return app


app = create_app()
