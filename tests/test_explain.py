"""Tests for prediction explainability. Requires trained model artifacts in
models/ (run src/evaluate.py first if missing)."""
import os
import sys
from datetime import datetime

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from explain import explain_by_deviation, explain_linear_regression, explain_prediction  # noqa: E402
from live_predict import build_feature_row  # noqa: E402

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
MODELS_MISSING = not os.path.exists(os.path.join(MODELS_DIR, "linear_regression.joblib"))

pytestmark = pytest.mark.skipif(MODELS_MISSING, reason="trained model artifacts not present")


def test_explain_linear_regression_returns_sorted_by_magnitude():
    row = build_feature_row(datetime(2026, 6, 15, 8, 0), 20.0, 0.0, 0.0, 10.0, 0, "Clear")
    ranked = explain_linear_regression(row, top_n=4)
    assert len(ranked) == 4
    magnitudes = [abs(d["contribution"]) for d in ranked]
    assert magnitudes == sorted(magnitudes, reverse=True)
    for d in ranked:
        assert "feature" in d and "label" in d and "contribution" in d


def test_explain_by_deviation_flags_extreme_rain_as_unusual():
    # This dataset's rain_1h is near-zero for the vast majority of hours, so a
    # heavy downpour should stand out with a large positive z-score.
    typical = build_feature_row(datetime(2026, 6, 15, 8, 0), 20.0, 0.0, 0.0, 10.0, 0, "Clear")
    stormy = build_feature_row(datetime(2026, 6, 15, 8, 0), 20.0, 50.0, 0.0, 10.0, 0, "Rain")

    typical_rain_z = next(d["z_score"] for d in explain_by_deviation(typical, top_n=17) if d["feature"] == "rain_1h")
    stormy_rain_z = next(d["z_score"] for d in explain_by_deviation(stormy, top_n=17) if d["feature"] == "rain_1h")
    assert stormy_rain_z > typical_rain_z
    assert stormy_rain_z > 3  # a genuinely extreme deviation, in standard deviations

    top = explain_by_deviation(stormy, top_n=4)
    assert "rain_1h" in [d["feature"] for d in top]


def test_explain_prediction_uses_exact_kind_for_linear_regression():
    row = build_feature_row(datetime(2026, 6, 15, 8, 0), 20.0, 0.0, 0.0, 10.0, 0, "Clear")
    result = explain_prediction("linear_regression", row)
    assert result["kind"] == "exact"
    assert len(result["top_features"]) == 4


def test_explain_prediction_uses_deviation_kind_for_non_linear_models():
    row = build_feature_row(datetime(2026, 6, 15, 8, 0), 20.0, 0.0, 0.0, 10.0, 0, "Clear")
    for model_name in ("svr", "lstm"):
        result = explain_prediction(model_name, row)
        assert result["kind"] == "deviation"
        assert len(result["top_features"]) == 4
