"""Tests for the CV/model fusion logic. The multiplicative-blend bug this project
shipped with (a real live density reading of 0.8 still produced "Low" congestion
because it barely nudged the historical baseline) is exactly the kind of thing
these tests exist to catch before it reaches a demo or a report."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cv_module import CONGESTION_LEVELS, congestion_from_density  # noqa: E402
from live_predict import fuse_labels  # noqa: E402


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
