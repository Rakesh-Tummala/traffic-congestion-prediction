"""Tests for the speed-estimation math and tracker — synthetic data only,
no live video required, so these run without network access."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from speed_estimation import (  # noqa: E402
    MPS_TO_KMH, MPS_TO_MPH, estimate_speed_mps, flag_speeding, track_across_frames,
)


def _det(x, y, cls_name="car"):
    return {"cls_name": cls_name, "conf": 0.9, "box": (x - 10, y - 10, x + 10, y + 10), "centroid": (x, y)}


def test_track_across_frames_follows_a_moving_vehicle():
    # A single car moving 20px/frame to the right across 4 frames.
    frame_detections = [[_det(100, 50)], [_det(120, 50)], [_det(140, 50)], [_det(160, 50)]]
    tracks = track_across_frames(frame_detections)
    assert len(tracks) == 1
    assert [d["centroid"] for d in tracks[0]] == [(100, 50), (120, 50), (140, 50), (160, 50)]


def test_track_across_frames_keeps_two_vehicles_separate():
    # Two cars far apart, both moving right — must not get crossed/merged.
    frame_detections = [
        [_det(100, 50), _det(100, 300)],
        [_det(120, 50), _det(125, 300)],
        [_det(140, 50), _det(150, 300)],
    ]
    tracks = track_across_frames(frame_detections)
    assert len(tracks) == 2
    ys = {round(t[0]["centroid"][1]) for t in tracks}
    assert ys == {50, 300}


def test_track_across_frames_ignores_distant_unrelated_detection():
    # A detection far outside max_match_dist_px must not get matched to an
    # unrelated vehicle. It starts its own track, which never reaches a
    # second point here, so only the real 2-point track is returned
    # (single-point tracks are dropped — no displacement to measure speed from).
    frame_detections = [[_det(100, 50)], [_det(105, 50)], [_det(900, 900)]]
    tracks = track_across_frames(frame_detections, max_match_dist_px=50)
    assert len(tracks) == 1
    assert [round(c) for c in tracks[0][-1]["centroid"]] == [105, 50]


def test_estimate_speed_mps_matches_known_displacement():
    # 100px in 1.0s at 0.1 m/px -> 10 m/s.
    track = [_det(0, 0), _det(100, 0)]
    track[0]["frame_idx"], track[1]["frame_idx"] = 0, 1
    speed = estimate_speed_mps(track, frame_times=[0.0, 1.0], meters_per_pixel=0.1)
    assert speed == 10.0


def test_estimate_speed_mps_none_for_single_point_track():
    track = [_det(0, 0)]
    track[0]["frame_idx"] = 0
    assert estimate_speed_mps(track, frame_times=[0.0], meters_per_pixel=0.1) is None


def test_mps_conversions_are_sane():
    assert abs(1.0 * MPS_TO_MPH - 2.23694) < 1e-3
    assert abs(1.0 * MPS_TO_KMH - 3.6) < 1e-9


def test_flag_speeding_respects_margin():
    # 4 mph over a 65 limit, with a 5 mph margin, should NOT be flagged.
    speeding, over = flag_speeding(69.0, 65.0, margin_mph=5.0)
    assert speeding is False
    assert over == 4.0

    # 10 mph over should be flagged.
    speeding, over = flag_speeding(75.0, 65.0, margin_mph=5.0)
    assert speeding is True
    assert over == 10.0
