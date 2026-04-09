"""Tests for FastAPI endpoints used for deployment."""

from __future__ import annotations

import pandas as pd
from fastapi.testclient import TestClient

from jane_street_portfolio.api import create_app
from jane_street_portfolio.service import JaneStreetModelService


def test_health_endpoint_returns_ok() -> None:
    """Health endpoint should return liveness information."""
    app = create_app(JaneStreetModelService())
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict_endpoint_returns_actions_when_model_loaded() -> None:
    """Predict endpoint should score data when a model is preloaded."""
    service = JaneStreetModelService()
    frame = pd.DataFrame(
        {
            "feature_0": [0.1, 0.2, 0.7, 0.8],
            "feature_1": [1.1, 1.2, 1.8, 1.9],
        }
    )
    target = pd.Series([0, 0, 1, 1])
    service.train(frame, target)

    app = create_app(service)
    client = TestClient(app)

    payload = {"rows": frame.to_dict(orient="records")}
    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    body = response.json()
    assert len(body["actions"]) == len(frame)
    assert len(body["probabilities"]) == len(frame)
