"""Tests for the live-inference feature pipeline, including the LSTM sequence path.
Requires trained model artifacts in models/ (run src/evaluate.py first if missing)."""
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from live_predict import ALL_MODELS, build_feature_row, predict_all_models  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODELS_MISSING = not all(
    os.path.exists(os.path.join(MODELS_DIR, f))
    for f in ("linear_regression.joblib", "svr.joblib", "lstm.pt", "lstm_scalers.joblib", "seasonal_profile.joblib")
)


def test_build_feature_row_has_no_missing_values():
    row = build_feature_row(datetime(2026, 6, 15, 8, 0), 20.0, 0.0, 0.0, 10.0, 0, "Clear")
    assert not row.isna().any().any()


@pytest.mark.skipif(MODELS_MISSING, reason="trained model artifacts not present")
def test_predict_all_models_returns_all_three():
    now = datetime(2026, 6, 15, 8, 0)  # a weekday morning rush hour
    predictions = predict_all_models(now, 20.0, 0.0, 0.0, 10.0, 0, "Clear")
    assert set(predictions.keys()) == set(ALL_MODELS)
    for volume in predictions.values():
        assert volume > 0  # traffic volume can't be negative


@pytest.mark.skipif(MODELS_MISSING, reason="trained model artifacts not present")
def test_predict_all_models_with_real_sensor_reading_differs_from_fallback():
    now = datetime(2026, 6, 15, 8, 0)
    baseline = predict_all_models(now, 20.0, 0.0, 0.0, 10.0, 0, "Clear")
    with_sensor = predict_all_models(now, 20.0, 0.0, 0.0, 10.0, 0, "Clear",
                                      lag_1h=100.0, lag_3h=100.0, lag_24h=100.0)
    # Supplying a real (very low) recent reading should pull every model's
    # prediction down relative to the seasonal-average fallback.
    for name in ALL_MODELS:
        assert with_sensor[name] < baseline[name]
