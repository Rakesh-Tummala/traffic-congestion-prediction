"""Tests for the route map helpers (network mocked — no live OSRM or Nominatim calls)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import math  # noqa: E402

from route_map import (  # noqa: E402
    cameras_near_route, color_route, fetch_route, fit_view, geocode, haversine_m, order_cameras,
)


def _cam(name, lat, lon):
    return {"name": name, "latitude": lat, "longitude": lon}


class _FakeResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):
        pass

    def json(self):
        return self._payload


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


def test_geocode_parses_results_and_sends_an_identifying_user_agent(monkeypatch):
    seen = {}

    def fake_get(url, params=None, headers=None, timeout=None):
        seen.update(params=params, headers=headers)
        return _FakeResponse([{"display_name": "Burbank Airport, CA", "lat": "34.2", "lon": "-118.35"}])

    monkeypatch.setattr("route_map.requests.get", fake_get)
    results = geocode("Burbank Airport")
    assert results == [{"name": "Burbank Airport, CA", "lat": 34.2, "lon": -118.35}]
    assert seen["params"]["q"] == "Burbank Airport"
    assert "traffic-congestion-prediction" in seen["headers"]["User-Agent"]


def test_geocode_blank_query_makes_no_request(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not call the network for a blank query")

    monkeypatch.setattr("route_map.requests.get", boom)
    assert geocode("   ") == []


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


def _east_west_route(n=11, step=0.01):
    return [(34.0, -118.00 + i * step) for i in range(n)]  # ~0.92 km between points at this latitude


def test_cameras_near_route_keeps_nearby_ones_in_route_order():
    route = _east_west_route()
    cams = [_cam("late", 34.001, -117.92), _cam("early", 34.001, -117.99), _cam("far_north", 34.5, -117.95),
            _cam("far_east", 34.0, -116.0)]
    assert [c["name"] for c in cameras_near_route(route, cams)] == ["early", "late"]


def test_cameras_near_route_thins_to_max_cameras_keeping_both_ends():
    route = _east_west_route(n=101, step=0.001)
    cams = [_cam(f"c{i}", 34.0005, -118.0 + i * 0.001) for i in range(101)]
    picked = cameras_near_route(route, cams, max_cameras=5)
    assert len(picked) == 5
    assert picked[0]["name"] == "c0" and picked[-1]["name"] == "c100"


def test_cameras_near_route_skips_cameras_with_bad_coordinates():
    route = _east_west_route()
    cams = [_cam("ok", 34.0, -117.95), _cam("none", None, None), {"name": "missing"}]
    assert [c["name"] for c in cameras_near_route(route, cams)] == ["ok"]


def test_cameras_near_route_empty_inputs():
    assert cameras_near_route([], [_cam("a", 34.0, -118.0)]) == []
    assert cameras_near_route(_east_west_route(), []) == []


def test_color_route_marks_near_far_and_unlabeled_stretches():
    # Only one camera, at the west end. ~0.92 km per point.
    route = _east_west_route(n=21)
    runs = color_route(route, [(34.0, -118.00, "High")], max_distance_m=1500, far_distance_m=6000)
    tiers = [(r["label"], r["near"]) for r in runs]
    assert tiers[0] == ("High", True)
    assert ("High", False) in tiers          # 1.5-6 km away: kept, but flagged as a weaker guess
    assert tiers[-1] == (None, False)        # beyond 6 km: uncolored


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
    assert len(runs) == 1 and runs[0]["label"] == "Moderate" and runs[0]["near"] and len(runs[0]["path"]) == 4


def _px_per_degree_lon(zoom):
    return 512 * 2 ** zoom / 360


def test_fit_view_centers_on_the_points():
    lat, lon, _ = fit_view([[-118.4, 34.0], [-118.2, 34.2]])
    assert lon == pytest.approx(-118.3)
    assert 34.0 < lat < 34.2


def test_fit_view_tall_route_is_limited_by_height_not_width():
    # ~0.2 deg wide but ~0.5 deg tall, in a wide short map: height is the binding limit.
    pts = [[-118.30, 34.00], [-118.10, 34.50]]
    _, _, zoom = fit_view(pts, width_px=1200, height_px=420)
    height_deg = 0.5
    assert height_deg * _px_per_degree_lon(zoom) / math.cos(math.radians(34.25)) <= 420  # fits vertically


def test_fit_view_wide_route_is_limited_by_width():
    pts = [[-119.0, 34.00], [-117.0, 34.05]]
    _, _, zoom = fit_view(pts, width_px=700, height_px=420)
    assert 2.0 * _px_per_degree_lon(zoom) <= 700  # fits horizontally


def test_fit_view_single_point_does_not_blow_up():
    lat, lon, zoom = fit_view([[-118.0, 34.0]])
    assert (lat, lon) == pytest.approx((34.0, -118.0), abs=1e-6)
    assert 1.0 <= zoom <= 15.0
