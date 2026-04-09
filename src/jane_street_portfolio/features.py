"""Feature engineering helpers for Jane Street style tabular data."""

from __future__ import annotations

import pandas as pd


def make_feature_matrix(frame: pd.DataFrame, feature_prefix: str = "feature_") -> pd.DataFrame:
    """Return a cleaned feature matrix using standard Jane Street feature columns.

    Args:
        frame: Input data containing model features and optional metadata columns.
        feature_prefix: Prefix used to identify model feature columns.

    Returns:
        A DataFrame containing only feature columns, with missing values filled by
        per-column medians.

    Raises:
        ValueError: If no columns match the provided feature prefix.
    """
    feature_cols = [col for col in frame.columns if col.startswith(feature_prefix)]
    if not feature_cols:
        raise ValueError(f"No columns found with prefix '{feature_prefix}'.")

    matrix = frame[feature_cols].copy()
    return matrix.fillna(matrix.median(numeric_only=True))
