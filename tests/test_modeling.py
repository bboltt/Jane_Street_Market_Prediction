"""Tests for baseline modeling helpers."""

import pandas as pd

from jane_street_portfolio.modeling import generate_actions, train_baseline_model


def test_train_baseline_model_and_generate_actions() -> None:
    """It should train a classifier and emit binary actions."""
    features = pd.DataFrame(
        {
            "feature_0": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
            "feature_1": [1.1, 1.2, 1.3, 1.4, 1.5, 1.6],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])

    result = train_baseline_model(features, target)
    actions = generate_actions(result, features, threshold=0.5)

    assert actions.shape[0] == features.shape[0]
    assert set(actions.tolist()).issubset({0, 1})
