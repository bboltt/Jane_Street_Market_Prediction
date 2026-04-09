"""Modeling utilities for a polished baseline Jane Street portfolio project."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


@dataclass
class BaselineModelResult:
    """Container for the trained baseline classifier and metadata."""

    model: LogisticRegression
    feature_names: list[str]


def train_baseline_model(features: pd.DataFrame, target: pd.Series) -> BaselineModelResult:
    """Train a logistic regression baseline for action prediction.

    Args:
        features: Numeric feature matrix.
        target: Binary target values, where 1 indicates a positive trade action.

    Returns:
        BaselineModelResult with fitted model and ordered feature names.
    """
    model = LogisticRegression(max_iter=500, n_jobs=None)
    model.fit(features, target)
    return BaselineModelResult(model=model, feature_names=features.columns.tolist())


def generate_actions(result: BaselineModelResult, features: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
    """Generate binary trading actions from a trained baseline model.

    Args:
        result: Trained model artifact.
        features: Feature matrix used for inference.
        threshold: Probability threshold for predicting an action.

    Returns:
        NumPy array of 0/1 action predictions.
    """
    ordered = features[result.feature_names]
    probabilities = result.model.predict_proba(ordered)[:, 1]
    return (probabilities >= threshold).astype(int)
