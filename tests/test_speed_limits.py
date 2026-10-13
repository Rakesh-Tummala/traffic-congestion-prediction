"""Tests for the OpenStreetMap speed-limit lookup (network mocked)."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import speed_limits as sl  # noqa: E402

LAT, LON = 34.0, -118.0


def way(highway, ref=None, name=None, maxspeed=None, offset=0.0):
    tags = {"highway": highway}
    if ref:
        tags["ref"] = ref
    if name:
        tags["name"] = name
    if maxspeed:
        tags["maxspeed"] = maxspeed
    # a short east-west segment `offset` degrees of latitude north of the camera
    return {"tags": tags, "geometry": [{"lat": LAT + offset, "lon": LON - 0.001}, {"lat": LAT + offset, "lon": LON + 0.001}]}


def test_parse_maxspeed_handles_units_and_non_numbers():
    assert sl.parse_maxspeed("65 mph") == 65
    assert sl.parse_maxspeed("65mph") == 65
    assert sl.parse_maxspeed("100") == pytest.approx(62.1)   # a bare number in OSM is km/h
    assert sl.parse_maxspeed("50 km/h") == pytest.approx(31.1)
    for bad in ("none", "signals", "variable", "", None, "walk"):
        assert sl.parse_maxspeed(bad) is None


def test_caltrans_route_names_map_to_osm_refs():
    assert sl.route_to_osm_refs("I-5") == {"I 5"}
    assert sl.route_to_osm_refs("US-101") == {"US 101"}
    assert sl.route_to_osm_refs("SR-134") == {"CA 134"}
    assert sl.route_to_osm_refs("") == set()
    assert sl.route_to_osm_refs("Main St") == set()


def test_camera_on_an_overpass_reports_the_freeway_not_the_street_crossing_it():
    street = way("primary", name="West Avenue 26", maxspeed="30 mph", offset=0.00001)  # closest
    freeway = way("motorway", ref="I 5", name="Santa Ana Freeway", maxspeed="65 mph", offset=0.0003)
    chosen = sl.pick_road([street, freeway], LAT, LON, route="I-5")
    assert chosen is freeway


def test_without_a_route_match_the_nearest_motorway_beats_a_nearer_street():
    street = way("residential", offset=0.00001)
    motorway = way("motorway", ref="CA 110;US 66 Hist", maxspeed="55 mph", offset=0.0004)
    assert sl.pick_road([street, motorway], LAT, LON, route="") is motorway


def test_posted_limit_is_reported_as_osm_and_missing_tag_as_assumed():
    posted = sl.limit_from_way(way("motorway", ref="I 5", name="Santa Ana Freeway", maxspeed="65 mph"))
    assert posted["limit_mph"] == 65 and posted["source"] == "osm" and "Santa Ana Freeway" in posted["detail"]
    assumed = sl.limit_from_way(way("motorway", ref="I 10"))
    assert assumed["limit_mph"] == 65 and assumed["source"] == "assumed" and "not a posted sign" in assumed["detail"]
    unknown = sl.limit_from_way(way("footway"))
    assert unknown["limit_mph"] is None and unknown["source"] == "none"


class _Resp:
    def __init__(self, status, payload=None):
        self.status_code, self._payload = status, payload or {}

    def json(self):
        return self._payload


def test_lookup_falls_back_to_the_second_server_and_caches(monkeypatch, tmp_path):
    sl._cache.clear()
    monkeypatch.setattr(sl, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(sl.time, "sleep", lambda s: None)
    calls = []

    def fake_post(url, data=None, timeout=None, headers=None):
        calls.append(url)
        if url == sl.OVERPASS_URLS[0]:
            return _Resp(504)
        return _Resp(200, {"elements": [way("motorway", ref="I 5", name="Freeway", maxspeed="65 mph")]})

    monkeypatch.setattr(sl.requests, "post", fake_post)
    first = sl.lookup_speed_limit(LAT, LON, "I-5")
    assert first["limit_mph"] == 65 and calls == sl.OVERPASS_URLS[:2]
    sl.lookup_speed_limit(LAT, LON, "I-5")
    assert len(calls) == 2  # second call came from the cache


def test_when_every_server_is_down_freeway_routes_get_a_provisional_fallback_that_is_not_cached(monkeypatch, tmp_path):
    sl._cache.clear()
    monkeypatch.setattr(sl, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(sl.time, "sleep", lambda s: None)
    monkeypatch.setattr(sl.requests, "post", lambda *a, **k: _Resp(504))
    got = sl.lookup_speed_limit(LAT, LON, "I-5")
    assert got["limit_mph"] == 65 and got["provisional"] and got["source"] == "assumed"
    assert "not this road's posted sign" in got["detail"]
    assert sl._cache == {} and not (tmp_path / "cache.json").exists()   # so the next call asks again
    assert sl.lookup_speed_limit(LAT, LON, "SR-134") is None             # no honest guess for other routes


def test_found_limits_are_remembered_on_disk_across_restarts(monkeypatch, tmp_path):
    sl._cache.clear()
    monkeypatch.setattr(sl, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(sl.time, "sleep", lambda s: None)
    monkeypatch.setattr(sl.requests, "post", lambda *a, **k: _Resp(200, {"elements": [
        way("motorway", ref="I 5", name="Freeway", maxspeed="65 mph")]}))
    assert sl.lookup_speed_limit(LAT, LON, "I-5")["limit_mph"] == 65
    sl._cache.clear()  # a fresh process: memory empty, and now the network is down
    monkeypatch.setattr(sl.requests, "post", lambda *a, **k: _Resp(504))
    again = sl.lookup_speed_limit(LAT, LON, "I-5")
    assert again["limit_mph"] == 65 and again["source"] == "osm" and not again.get("provisional")


def test_lookup_sends_an_identifying_user_agent(monkeypatch, tmp_path):
    sl._cache.clear()
    monkeypatch.setattr(sl, "CACHE_PATH", str(tmp_path / "cache.json"))
    seen = {}

    def fake_post(url, data=None, timeout=None, headers=None):
        seen.update(headers or {})
        return _Resp(200, {"elements": []})

    monkeypatch.setattr(sl.requests, "post", fake_post)
    result = sl.lookup_speed_limit(LAT, LON)
    assert "traffic-congestion-prediction" in seen["User-Agent"]
    assert result["limit_mph"] is None and result["source"] == "none"


def test_lookup_async_never_blocks_and_delivers_the_result_when_ready(monkeypatch, tmp_path):
    import threading
    sl._cache.clear(); sl._pending.clear(); sl._retry_after.clear()
    monkeypatch.setattr(sl, "CACHE_PATH", str(tmp_path / "cache.json"))
    gate = threading.Event()

    def slow_post(*a, **k):
        gate.wait(5)
        return _Resp(200, {"elements": [way("motorway", ref="I 5", name="Freeway", maxspeed="65 mph")]})

    monkeypatch.setattr(sl.requests, "post", slow_post)
    first = sl.lookup_async(LAT, LON, "I-5")
    assert first["state"] == "pending" and first["result"]["provisional"]   # instant, with a stand-in value
    gate.set()
    sl._pending[(round(LAT, 4), round(LON, 4), "I-5")].result(timeout=5)
    done = sl.lookup_async(LAT, LON, "I-5")
    assert done["state"] == "done" and done["result"]["source"] == "osm" and done["result"]["limit_mph"] == 65


def test_a_failed_async_lookup_is_not_retried_for_a_while(monkeypatch, tmp_path):
    sl._cache.clear(); sl._pending.clear(); sl._retry_after.clear()
    monkeypatch.setattr(sl, "CACHE_PATH", str(tmp_path / "cache.json"))
    monkeypatch.setattr(sl.time, "sleep", lambda s: None)
    calls = []
    monkeypatch.setattr(sl.requests, "post", lambda *a, **k: calls.append(1) or _Resp(504))
    sl.lookup_async(LAT, LON, "I-5")
    sl._pending[(round(LAT, 4), round(LON, 4), "I-5")].result(timeout=5)
    n = len(calls)
    out = sl.lookup_async(LAT, LON, "I-5")           # collects the failure
    again = sl.lookup_async(LAT, LON, "I-5")         # inside the retry window: no new request
    assert out["state"] == again["state"] == "done" and again["result"]["provisional"]
    assert len(calls) == n
