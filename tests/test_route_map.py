"""Tests for the route map helpers (network mocked — no live OSRM calls)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from route_map import color_route, fetch_route, haversine_m, order_cameras  # noqa: E402


def _cam(name, lat, lon):
    return {"name": name, "latitude": lat, "longitude": lon}


def test_haversine_one_degree_of_latitude_is_about_111km():
    assert haversine_m(34.0, -118.0, 35.0, -118.0) == pytest.approx(111_195, rel=0.01)


def test_haversine_same_point_is_zero():
    assert haversine_m(34.1, -118.2, 34.1, -118.2) == 0


def test_order_cameras_puts_a_shuffled_corridor_in_order_west_to_east():
    cams = [_cam("c", 34.00, -117.90), _cam("a", 34.00, -118.30), _cam("d", 34.00, -117.70),
            _cam("b", 34.00, -118.10)]
    assert [c["name"] for c in order_cameras(cams)] == ["a", "b", "c", "d"]


def test_order_cameras_handles_string_coordinates_and_two_cameras():
    cams = [_cam("east", "34.0", "-117.0"), _cam("west", "34.0", "-118.0")]
    assert [c["name"] for c in order_cameras(cams)] == ["west", "east"]


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


def test_fetch_route_parses_geometry_distance_and_duration(monkeypatch):
    seen = {}

    def fake_get(url, params=None, timeout=None):
        seen["url"], seen["params"] = url, params
        return _FakeResponse({"code": "Ok", "routes": [{
            "distance": 5000.0, "duration": 300.0,
            "geometry": {"coordinates": [[-118.0, 34.0], [-118.1, 34.05]]},
        }]})

    monkeypatch.setattr("route_map.requests.get", fake_get)
    route = fetch_route((34.0, -118.0), (34.05, -118.1))
    assert "-118.0,34.0;-118.1,34.05" in seen["url"]  # OSRM wants lon,lat order
    assert route["points"] == [(34.0, -118.0), (34.05, -118.1)]  # converted back to lat, lon
    assert route["distance_m"] == 5000.0 and route["duration_s"] == 300.0


def test_fetch_route_raises_when_osrm_finds_no_route(monkeypatch):
    monkeypatch.setattr("route_map.requests.get",
                        lambda *a, **k: _FakeResponse({"code": "NoRoute", "routes": []}))
    with pytest.raises(ValueError):
        fetch_route((0.0, 0.0), (1.0, 1.0))


def test_color_route_uses_nearest_camera_and_leaves_far_stretches_unlabeled():
    # A straight east-west road; cameras only near the two ends.
    route = [(34.0, -118.00 + i * 0.01) for i in range(11)]  # ~0.92 km between points
    readings = [(34.0, -118.00, "High"), (34.0, -117.90, "Low")]
    runs = color_route(route, readings, max_distance_m=1500)
    labels = [r["label"] for r in runs]
    assert labels[0] == "High" and labels[-1] == "Low"
    assert None in labels  # the middle is more than 1.5km from either camera


def test_color_route_runs_are_continuous_with_no_gaps():
    route = [(34.0, -118.00 + i * 0.001) for i in range(6)]
    readings = [(34.0, -118.00, "Low"), (34.0, -117.995, "Severe")]
    runs = color_route(route, readings, max_distance_m=5000)
    assert len(runs) >= 2
    for prev, nxt in zip(runs, runs[1:]):
        assert prev["path"][-1] == nxt["path"][0]  # next run starts where the last ended


def test_color_route_single_label_gives_one_run():
    route = [(34.0, -118.0 + i * 0.001) for i in range(4)]
    runs = color_route(route, [(34.0, -118.0, "Moderate")], max_distance_m=5000)
    assert len(runs) == 1 and runs[0]["label"] == "Moderate" and len(runs[0]["path"]) == 4
