"""Tests for vehicle color/body-type identification."""
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vehicle_attributes import (  # noqa: E402
    classify_vehicle_type, color_name, dominant_color_bgr,
)


def _solid_frame(bgr, size=60):
    frame = np.zeros((size, size, 3), dtype=np.uint8)
    frame[:, :] = bgr
    return frame


def test_dominant_color_bgr_reads_solid_mid_brightness_patch():
    frame = _solid_frame((200, 30, 30))  # BGR: strong blue patch
    rgb = dominant_color_bgr(frame, (0, 0, 60, 60))
    assert rgb is not None
    r, g, b = rgb
    assert b > r and b > g


def test_dominant_color_bgr_empty_box_returns_none():
    frame = _solid_frame((200, 30, 30))
    assert dominant_color_bgr(frame, (10, 10, 10, 10)) is None


def test_dominant_color_bgr_out_of_bounds_box_clips_to_frame():
    frame = _solid_frame((0, 0, 0))
    rgb = dominant_color_bgr(frame, (-50, -50, 200, 200))
    assert rgb is not None


def test_dominant_color_bgr_falls_back_when_all_pixels_too_dark():
    frame = _solid_frame((5, 5, 5))  # below the V>60 filter for every pixel
    rgb = dominant_color_bgr(frame, (0, 0, 60, 60))
    assert rgb is not None  # falls back to unfiltered pixels instead of returning None


@pytest.mark.parametrize("rgb,expected", [
    ((10, 10, 10), "black"),
    ((230, 230, 230), "white"),
    ((160, 160, 160), "silver"),
    ((90, 90, 90), "gray"),
    ((200, 30, 30), "red"),
    ((30, 60, 200), "blue"),
    ((40, 180, 40), "green"),
])
def test_color_name_thresholds(rgb, expected):
    assert color_name(rgb) == expected


def test_classify_vehicle_type_empty_box_returns_none_without_loading_model():
    frame = _solid_frame((100, 100, 100))
    result = classify_vehicle_type(frame, (10, 10, 10, 10))
    assert result == {"type": None, "confidence": 0.0}


@pytest.mark.slow
def test_classify_vehicle_type_runs_on_real_sample_and_returns_a_known_label():
    import cv2
    sample_path = os.path.join(os.path.dirname(__file__), "..", "static", "samples", "traffic_jam.jpg")
    frame = cv2.imread(sample_path)
    assert frame is not None
    h, w = frame.shape[:2]
    result = classify_vehicle_type(frame, (0, 0, w, h), min_confidence=0.0)
    assert 0.0 <= result["confidence"] <= 1.0
    assert result["type"] in {label for _, label in
                              __import__("vehicle_attributes").VEHICLE_TYPE_PROMPTS}
