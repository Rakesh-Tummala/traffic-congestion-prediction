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


def test_tile_windows_cover_the_whole_frame_with_overlap():
    from cv_module import tile_windows
    wins = tile_windows(720, 1280, (2, 2), overlap=0.2)
    assert len(wins) == 4
    assert min(w[0] for w in wins) == 0 and max(w[2] for w in wins) == 1280
    assert min(w[1] for w in wins) == 0 and max(w[3] for w in wins) == 720
    left, right = wins[0], wins[1]
    assert right[0] < left[2]  # neighbours overlap, so a car on the seam is whole in one of them


def test_merge_boxes_drops_a_car_found_twice_and_the_fragment_cut_by_a_tile_edge():
    from cv_module import merge_boxes
    full = {"box": (100, 100, 200, 160), "conf": 0.9, "cls_id": 2}
    dup = {"box": (102, 101, 199, 161), "conf": 0.6, "cls_id": 2}      # same car, second window
    fragment = {"box": (100, 100, 130, 160), "conf": 0.5, "cls_id": 2}  # left sliver of it, cut by a tile edge
    other = {"box": (400, 100, 500, 160), "conf": 0.7, "cls_id": 2}
    kept = merge_boxes([dup, other, fragment, full])
    assert full in kept and other in kept and len(kept) == 2


def test_merge_boxes_keeps_overlapping_boxes_of_different_classes():
    from cv_module import merge_boxes
    car = {"box": (100, 100, 200, 160), "conf": 0.9, "cls_id": 2}
    truck = {"box": (100, 100, 200, 160), "conf": 0.8, "cls_id": 7}
    assert len(merge_boxes([car, truck])) == 2
