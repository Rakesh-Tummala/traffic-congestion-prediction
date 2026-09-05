"""Tests for dataset loading and feature engineering."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import FEATURE_COLUMNS, LAG_COLUMNS, TARGET_COLUMN, load_dataset  # noqa: E402


@pytest.fixture(scope="module")
def dataset():
    return load_dataset()


def test_load_dataset_has_no_missing_features(dataset):
    df, X, y = dataset
    assert not X.isna().any().any()
    assert not y.isna().any()


def test_feature_columns_present(dataset):
    df, X, y = dataset
    assert list(X.columns) == FEATURE_COLUMNS
    for col in LAG_COLUMNS:
        assert col in df.columns


def test_cyclical_hour_encoding_wraps_around(dataset):
    df, _, _ = dataset
    hour_23 = df[df["hour"] == 23].iloc[0]
    hour_0 = df[df["hour"] == 0].iloc[0]
    # Hour 23 and hour 0 are adjacent on the clock, so their encodings should
    # be close together, not far apart the way raw integers 23 and 0 would be.
    dist = np.hypot(hour_23["hour_sin"] - hour_0["hour_sin"], hour_23["hour_cos"] - hour_0["hour_cos"])
    assert dist < 1.0


def test_congestion_levels_roughly_balanced(dataset):
    df, _, _ = dataset
    counts = df["congestion_level"].value_counts()
    assert len(counts) == 4
    # Quartile-based bins should be roughly equal-sized (within 5%).
    assert counts.min() / counts.max() > 0.9


def test_lag_1h_correlates_with_target(dataset):
    df, _, _ = dataset
    # A lag feature that isn't at least moderately correlated with the target
    # would indicate the reindex/shift logic is broken.
    corr = df["lag_1h"].corr(df[TARGET_COLUMN])
    assert corr > 0.7
