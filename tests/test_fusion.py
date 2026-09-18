"""Tests for the CV/model fusion logic. The multiplicative-blend bug this project
shipped with (a real live density reading of 0.8 still produced "Low" congestion
because it barely nudged the historical baseline) is exactly the kind of thing
these tests exist to catch before it reaches a demo or a report."""
import os
import sys

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cv_module import CONGESTION_LEVELS, congestion_from_density  # noqa: E402
from live_predict import congestion_mismatch, detect_anomaly, fuse_labels, recommend_departure  # noqa: E402


def test_congestion_from_density_thresholds():
    assert congestion_from_density(0.0) == "Low"
    assert congestion_from_density(0.049) == "Low"
    assert congestion_from_density(0.05) == "Moderate"
    assert congestion_from_density(0.15) == "High"
    assert congestion_from_density(0.30) == "Severe"
    assert congestion_from_density(0.99) == "Severe"


def test_fuse_labels_takes_more_severe_signal():
    # A gridlocked live camera must never be reported as low congestion just
    # because the historical time/weather model expects a quiet hour.
    assert fuse_labels("Low", "Severe") == "Severe"
    assert fuse_labels("Severe", "Low") == "Severe"
    assert fuse_labels("Moderate", "High") == "High"
    assert fuse_labels("High", "Moderate") == "High"


def test_fuse_labels_identity():
    for level in CONGESTION_LEVELS:
        assert fuse_labels(level, level) == level


def test_congestion_levels_are_ordered_low_to_severe():
    assert CONGESTION_LEVELS == ["Low", "Moderate", "High", "Severe"]


def test_congestion_mismatch_is_symmetric_severity_gap():
    assert congestion_mismatch("Low", "Low") == 0
    assert congestion_mismatch("Low", "Severe") == 3
    assert congestion_mismatch("Severe", "Low") == 3
    assert congestion_mismatch("Moderate", "High") == 1


def test_detect_anomaly_flags_large_mismatch_only():
    assert detect_anomaly("Low", "Severe") is True
    assert detect_anomaly("Low", "High") is True
    assert detect_anomaly("Low", "Moderate") is False  # one-level wobble, not an anomaly
    assert detect_anomaly("Moderate", "Moderate") is False


def test_detect_anomaly_respects_custom_threshold():
    assert detect_anomaly("Low", "Moderate", threshold=1) is True
    assert detect_anomaly("Low", "Moderate", threshold=2) is False


def _forecast_df(hour_to_volume: dict) -> pd.DataFrame:
    labels = {v: k for k, v in enumerate(CONGESTION_LEVELS)}
    rows = []
    for hour, volume in hour_to_volume.items():
        level = CONGESTION_LEVELS[min(3, int(volume // 1000))]
        rows.append({"hour": hour, "volume": volume, "label": level})
    return pd.DataFrame(rows)


def test_recommend_departure_finds_lower_congestion_in_window():
    # Target hour (17) is jammed; hour 15 nearby is much clearer.
    df = _forecast_df({15: 500, 16: 2500, 17: 3800, 18: 3200, 19: 1200})
    rec = recommend_departure(df, target_hour=17, flexibility_hours=2)
    assert rec["target_label"] == "Severe"
    assert rec["best_hour"] == 15
    assert rec["best_label"] == "Low"
    assert rec["improves"] is True


def test_recommend_departure_no_improvement_when_target_is_already_best():
    df = _forecast_df({15: 3000, 16: 2800, 17: 500, 18: 2900, 19: 3100})
    rec = recommend_departure(df, target_hour=17, flexibility_hours=2)
    assert rec["best_hour"] == 17
    assert rec["improves"] is False


def test_recommend_departure_window_wraps_around_midnight():
    df = _forecast_df({h: 3000 for h in range(24)})
    df.loc[df["hour"] == 0, "volume"] = 200
    df.loc[df["hour"] == 0, "label"] = "Low"
    rec = recommend_departure(df, target_hour=23, flexibility_hours=1)
    assert rec["best_hour"] == 0
    assert rec["improves"] is True
