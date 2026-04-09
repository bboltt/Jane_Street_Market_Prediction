"""Deployment-oriented service primitives for model training and inference."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from .config import ModelConfig
from .features import make_feature_matrix


@dataclass
class PredictionResult:
    """Represents scored probabilities and thresholded binary actions."""

    probabilities: np.ndarray
    actions: np.ndarray


class JaneStreetModelService:
    """High-level service API for training, persistence, and prediction.

    The class wraps a scikit-learn pipeline so the repository can be deployed as a
    repeatable service rather than a notebook-only experiment.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """Initialize service with runtime configuration."""
        self.config = config or ModelConfig()
        self.pipeline: Pipeline | None = None
        self.feature_names: list[str] = []

    def train(self, frame: pd.DataFrame, target: pd.Series) -> None:
        """Train a baseline logistic pipeline from an input frame and labels."""
        features = make_feature_matrix(frame)
        self.feature_names = features.columns.tolist()
        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("classifier", LogisticRegression(max_iter=500)),
            ]
        )
        self.pipeline.fit(features, target)

    def predict(self, frame: pd.DataFrame) -> PredictionResult:
        """Predict action probabilities and binary actions for incoming rows.

        Raises:
            RuntimeError: If prediction is attempted before a model is loaded.
        """
        if self.pipeline is None:
            raise RuntimeError("Model is not loaded. Train or load an artifact first.")

        if self.feature_names:
            matrix = frame[self.feature_names]
        else:
            matrix = make_feature_matrix(frame)

        probabilities = self.pipeline.predict_proba(matrix)[:, 1]
        actions = (probabilities >= self.config.decision_threshold).astype(int)
        return PredictionResult(probabilities=probabilities, actions=actions)

    def save(self, path: Path | None = None) -> Path:
        """Persist the trained model pipeline and metadata to disk."""
        if self.pipeline is None:
            raise RuntimeError("Cannot save before model is trained.")

        output_path = path or self.config.model_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"pipeline": self.pipeline, "feature_names": self.feature_names}
        joblib.dump(payload, output_path)
        return output_path

    def load(self, path: Path | None = None) -> None:
        """Load serialized model pipeline and metadata from disk."""
        model_path = path or self.config.model_path
        payload = joblib.load(model_path)
        self.pipeline = payload["pipeline"]
        self.feature_names = payload.get("feature_names", [])
