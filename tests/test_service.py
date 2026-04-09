"""Tests for deployment service behavior and model artifact lifecycle."""

from __future__ import annotations

from pathlib import Path
import tempfile

import pandas as pd

from jane_street_portfolio.service import JaneStreetModelService


def test_service_train_predict_save_and_load_round_trip() -> None:
    """Model service should support end-to-end train, predict, save, and load."""
    frame = pd.DataFrame(
        {
            "feature_0": [0.1, 0.2, 0.3, 0.8, 0.9, 1.0],
            "feature_1": [1.0, 1.1, 1.2, 1.7, 1.8, 1.9],
        }
    )
    target = pd.Series([0, 0, 0, 1, 1, 1])

    service = JaneStreetModelService()
    service.train(frame, target)

    result = service.predict(frame)
    assert len(result.probabilities) == len(frame)
    assert set(result.actions.tolist()).issubset({0, 1})

    with tempfile.TemporaryDirectory() as tmp_dir:
        model_path = Path(tmp_dir) / "model.joblib"
        service.save(path=model_path)

        restored = JaneStreetModelService()
        restored.load(model_path)
        restored_result = restored.predict(frame)

    assert len(restored_result.actions) == len(frame)
