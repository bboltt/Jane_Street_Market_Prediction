"""Application configuration for training and inference services."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Configuration values used by the model service layer.

    Attributes:
        model_path: Filesystem path where serialized model artifacts are stored.
        decision_threshold: Probability cut-off for converting scores to actions.
    """

    model_path: Path = Path("artifacts/baseline_model.joblib")
    decision_threshold: float = 0.5
