"""Tests for the frame-validity check in cv_module (no model is loaded)."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from cv_module import CameraUnavailableError, analyze_frame, is_unavailable_frame  # noqa: E402


def _placeholder(h=240, w=320):
    frame = np.full((h, w, 3), 255, dtype=np.uint8)
    cv2.putText(frame, "Temporarily", (30, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 0, 0), 3)
    cv2.putText(frame, "Unavailable", (30, 150), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (120, 0, 0), 3)
    frame[-30:] = 90  # the grey timestamp bar Caltrans adds under the image
    return frame


def test_placeholder_image_is_detected_as_unavailable():
    assert is_unavailable_frame(_placeholder())


def test_ordinary_road_frame_is_not_flagged():
    rng = np.random.default_rng(0)
    road = rng.integers(60, 160, size=(240, 320, 3), dtype=np.uint8)
    assert not is_unavailable_frame(road)


def test_bright_but_real_frame_with_some_white_is_not_flagged():
    frame = np.full((240, 320, 3), 120, dtype=np.uint8)
    frame[:60] = 255  # a washed-out sky: 25% white, like the brightest real frames measured
    assert not is_unavailable_frame(frame)


def test_empty_or_missing_frame_is_not_flagged():
    assert not is_unavailable_frame(None)
    assert not is_unavailable_frame(np.zeros((0, 0, 3), dtype=np.uint8))


def test_analyze_frame_refuses_the_placeholder_instead_of_reporting_an_empty_road():
    with pytest.raises(CameraUnavailableError):
        analyze_frame(_placeholder())
