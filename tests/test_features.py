"""Tests for feature engineering utilities."""

import pandas as pd
import pytest

from jane_street_portfolio.features import make_feature_matrix


def test_make_feature_matrix_selects_and_imputes_features() -> None:
    """It should keep feature columns and impute missing values with medians."""
    frame = pd.DataFrame(
        {
            "feature_0": [1.0, None, 3.0],
            "feature_1": [2.0, 4.0, None],
            "date": [1, 1, 2],
        }
    )

    matrix = make_feature_matrix(frame)

    assert list(matrix.columns) == ["feature_0", "feature_1"]
    assert matrix.isna().sum().sum() == 0
    assert matrix.loc[1, "feature_0"] == 2.0


def test_make_feature_matrix_raises_when_no_matching_columns() -> None:
    """It should raise ValueError when no feature-prefixed columns exist."""
    frame = pd.DataFrame({"x": [1, 2, 3]})

    with pytest.raises(ValueError, match="No columns found"):
        make_feature_matrix(frame)
