"""Tests for tracking, camera calibration and speed estimation, using synthetic cars of known speed."""
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from live_monitor import SCALE_RANGE, speed_scale, violation_threshold  # noqa: E402
from vehicle_tracking import (  # noqa: E402
    CAR_WIDTH_M, CalibrationFitter, CentroidTracker, GroundCalibration, MPS_TO_MPH, estimate_speed_mph,
)

W, H = 1280, 720
CALIB = GroundCalibration(horizon_y=100.0, camera_height_m=7.0, focal_px=1100.0, image_width=W,
                          correlation=0.95, n_samples=100)


def det_at(z: float, x_m: float = 0.0, cls="car", conf=0.9) -> dict:
    """Detection of a vehicle at ground position (x_m sideways, z forward) seen by CALIB's camera."""
    y_bottom = CALIB.horizon_y + CALIB.focal_px * CALIB.camera_height_m / z
    width = CALIB.focal_px * CAR_WIDTH_M / z
    xc = W / 2 + CALIB.focal_px * x_m / z
    return {"cls_name": cls, "conf": conf, "box": (xc - width / 2, y_bottom - width * 0.8, xc + width / 2, y_bottom)}


def drive(speed_mps: float, seconds=3.0, dt=0.27, z0=115.0, x_m=2.0) -> CentroidTracker:
    tracker = CentroidTracker()
    t = 0.0
    while t <= seconds:
        tracker.update(t, [det_at(z0 - speed_mps * t, x_m)])  # approaching the camera
        t += dt
    return tracker


def test_speed_estimate_matches_the_true_speed_of_a_synthetic_car():
    tracker = drive(speed_mps=25.0)  # 25 m/s = 55.9 mph
    assert len(tracker.tracks) == 1
    mph = estimate_speed_mph(tracker.tracks[0], CALIB)
    assert mph == pytest.approx(25.0 * MPS_TO_MPH, rel=0.04)


def test_faster_car_reads_faster():
    slow = estimate_speed_mph(drive(15.0).tracks[0], CALIB)
    fast = estimate_speed_mph(drive(30.0).tracks[0], CALIB)
    assert fast > slow * 1.8


def test_too_little_history_gives_no_speed():
    tracker = CentroidTracker()
    tracker.update(0.0, [det_at(60.0)])
    tracker.update(0.27, [det_at(59.0)])
    assert estimate_speed_mph(tracker.tracks[0], CALIB) is None


def test_tracker_keeps_one_id_per_vehicle_and_separates_two_cars():
    tracker = CentroidTracker()
    ids_seen = set()
    for i in range(8):
        t = i * 0.27
        tracked = tracker.update(t, [det_at(120 - 20 * t, -3.0), det_at(100 - 25 * t, 3.5)])
        ids_seen |= {tr.track_id for tr in tracked}
    assert ids_seen == {1, 2}


def test_tracker_drops_a_vehicle_after_it_disappears():
    tracker = CentroidTracker(max_missed=2)
    tracker.update(0.0, [det_at(60.0)])
    for i in range(1, 5):
        tracker.update(i * 0.27, [])
    assert tracker.tracks == []


def _cars_on_a_flat_road(n=200, seed=0, height=7.0, noise=2.0):
    rng = np.random.default_rng(seed)
    fitter = CalibrationFitter(W, H)
    f = (W / 2) / math.tan(math.radians(31))
    for z in rng.uniform(15, 110, n):
        width = f * CAR_WIDTH_M / z
        y_bottom = 100.0 + f * height / z + rng.normal(0, noise)
        fitter.samples.append((width, y_bottom))
    return fitter


def test_calibration_recovers_camera_height_from_cars_at_many_distances():
    cal = _cars_on_a_flat_road(height=7.0).fit()
    assert cal is not None
    assert cal.camera_height_m == pytest.approx(7.0, rel=0.1)
    assert cal.horizon_y == pytest.approx(100.0, abs=15)


def test_calibration_refuses_too_few_cars_or_no_trend():
    few = CalibrationFitter(W, H)
    few.samples = [(30.0, 300.0)] * 10
    assert few.fit() is None
    rng = np.random.default_rng(1)
    noise = CalibrationFitter(W, H)
    noise.samples = [(float(w), float(y)) for w, y in zip(rng.uniform(20, 80, 200), rng.uniform(100, 600, 200))]
    assert noise.fit() is None  # width and row are unrelated: not a flat road seen along its length


def test_to_ground_returns_none_at_or_above_the_horizon():
    assert CALIB.to_ground(640, 100.0) is None
    assert CALIB.to_ground(640, 50.0) is None


def test_speed_scale_matches_the_median_to_the_reference_and_rejects_big_corrections():
    assert speed_scale([20.0] * 15, 60.0) == pytest.approx(3.0) or speed_scale([20.0] * 15, 60.0) is None
    assert speed_scale([40.0] * 15, 60.0) == pytest.approx(1.5)
    assert speed_scale([10.0] * 15, 60.0) is None          # would need 6x: camera model unusable
    assert speed_scale([40.0] * 5, 60.0) is None           # too few tracked vehicles
    lo, hi = SCALE_RANGE
    assert lo < 1.0 < hi


def test_violation_threshold_is_limit_plus_margin():
    assert violation_threshold(65, 15) == 80
