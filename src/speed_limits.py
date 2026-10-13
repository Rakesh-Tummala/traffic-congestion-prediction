"""Find the posted speed limit for a camera's location from OpenStreetMap.

OpenStreetMap stores a `maxspeed` tag on many roads (free, no key; Overpass API). Given a
camera's coordinates and its route name (e.g. "I-5"), this finds the road the camera is
watching and reads that tag. Order of preference:

1. a nearby road whose `ref` matches the camera's route (so a camera on an overpass reports the
   freeway, not the street crossing it), nearest first;
2. otherwise the nearest motorway, then the nearest road of any kind.

If the chosen road has no `maxspeed` tag, a typical default for that road class is returned and
clearly marked as assumed rather than posted. If the service cannot be reached, None is
returned and the caller keeps its manual value. Results are cached per location.
"""
import json
import math
import os
import re
import time
from concurrent.futures import Future, ThreadPoolExecutor

import requests

from route_map import USER_AGENT

OVERPASS_URLS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
                 "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
CACHE_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "speed_limits_cache.json")
FALLBACK_FREEWAY_MPH = 65.0  # California's usual limit on Interstates / US highways when nothing is posted
SEARCH_RADIUS_M = 60
KMH_TO_MPH = 0.621371
# Typical limits when a road has no maxspeed tag (California, mph). Assumed, not posted.
DEFAULT_MPH = {"motorway": 65.0, "motorway_link": 45.0, "trunk": 55.0, "trunk_link": 35.0,
               "primary": 45.0, "secondary": 40.0, "tertiary": 35.0, "residential": 25.0}
_cache: dict = {}


def parse_maxspeed(value: str | None) -> float | None:
    """OSM maxspeed text to mph. '65 mph' -> 65; a bare number is km/h in OSM, so '100' -> 62;
    'none', 'signals', 'variable' and other non-numbers -> None."""
    if not value:
        return None
    m = re.match(r"^\s*(\d+(?:\.\d+)?)\s*(mph|km/h|kmh)?\s*$", value.strip().lower())
    if not m:
        return None
    number, unit = float(m.group(1)), m.group(2)
    return number if unit == "mph" else round(number * KMH_TO_MPH, 1)


def route_to_osm_refs(route: str) -> set[str]:
    """Caltrans route name to the OSM `ref` spellings it may appear as: 'I-5' -> {'I 5'},
    'US-101' -> {'US 101'}, 'SR-134' -> {'CA 134'}."""
    m = re.match(r"^\s*(I|US|SR|CA|HWY)[\s\-]*(\d+)", (route or "").upper())
    if not m:
        return set()
    kind, num = m.group(1), m.group(2)
    return {{"I": f"I {num}", "US": f"US {num}", "SR": f"CA {num}", "CA": f"CA {num}", "HWY": f"CA {num}"}[kind]}


def _distance_to_way_m(lat: float, lon: float, geometry: list[dict]) -> float:
    """Distance from a point to a polyline, using a local flat-earth projection (fine at 60 m)."""
    kx = 111_320 * math.cos(math.radians(lat))
    ky = 110_540
    pts = [((g["lon"] - lon) * kx, (g["lat"] - lat) * ky) for g in geometry]
    best = float("inf")
    for (x1, y1), (x2, y2) in zip(pts, pts[1:]):
        dx, dy = x2 - x1, y2 - y1
        seg = dx * dx + dy * dy
        t = 0.0 if seg == 0 else max(0.0, min(1.0, -(x1 * dx + y1 * dy) / seg))
        best = min(best, math.hypot(x1 + t * dx, y1 + t * dy))
    return best if pts[1:] else (math.hypot(*pts[0]) if pts else float("inf"))


def pick_road(ways: list[dict], lat: float, lon: float, route: str = "") -> dict | None:
    """The way the camera is most likely watching (see module docstring for the ranking)."""
    refs = route_to_osm_refs(route)
    scored = []
    for w in ways:
        tags = w.get("tags", {})
        if not w.get("geometry") or "highway" not in tags:
            continue
        way_refs = {r.strip() for r in tags.get("ref", "").split(";") if r.strip()}
        matches = bool(refs & way_refs)
        is_motorway = tags["highway"] == "motorway"
        rank = 0 if matches else 1 if is_motorway else 2
        scored.append((rank, _distance_to_way_m(lat, lon, w["geometry"]), w))
    if not scored:
        return None
    return min(scored, key=lambda s: (s[0], s[1]))[2]


def limit_from_way(way: dict) -> dict:
    tags = way["tags"]
    road = tags.get("name") or tags.get("ref") or tags["highway"]
    if tags.get("ref") and tags.get("name"):
        road = f"{tags['name']} ({tags['ref'].split(';')[0]})"
    posted = parse_maxspeed(tags.get("maxspeed"))
    if posted is not None:
        return {"limit_mph": round(posted), "source": "osm",
                "detail": f"Posted limit from OpenStreetMap for {road}: {tags['maxspeed']}."}
    default = DEFAULT_MPH.get(tags["highway"])
    if default is None:
        return {"limit_mph": None, "source": "none",
                "detail": f"{road} has no speed limit in OpenStreetMap."}
    return {"limit_mph": default, "source": "assumed",
            "detail": f"{road} has no posted limit in OpenStreetMap, so this is a typical value for a "
                      f"{tags['highway'].replace('_', ' ')} road, not a posted sign."}


def _query_overpass(lat: float, lon: float, timeout: float, rounds: int = 2) -> list[dict] | None:
    """Ask each Overpass mirror in turn (the public servers are often busy: 504s are normal)."""
    query = f'[out:json][timeout:20];way(around:{SEARCH_RADIUS_M},{lat},{lon})["highway"];out tags geom;'
    for _ in range(rounds):
        for url in OVERPASS_URLS:
            try:
                resp = requests.post(url, data={"data": query}, timeout=timeout,
                                     headers={"User-Agent": USER_AGENT, "Accept": "application/json"})
                if resp.status_code == 200:
                    return resp.json().get("elements", [])
            except (requests.RequestException, ValueError):
                pass
            time.sleep(0.5)
    return None


def _load_disk_cache() -> dict:
    try:
        with open(CACHE_PATH, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _save_disk_cache(key: str, value: dict) -> None:
    try:
        data = _load_disk_cache()
        data[key] = value
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        pass  # a cache that cannot be written just means looking it up again next time


def fallback_for_route(route: str) -> dict | None:
    """What to show when OpenStreetMap cannot be reached: the usual California freeway limit for an
    Interstate or US route, clearly marked as provisional. None for other routes (no honest guess)."""
    if re.match(r"^\s*(I|US)[\s\-]*\d+", (route or "").upper()):
        return {"limit_mph": FALLBACK_FREEWAY_MPH, "source": "assumed", "provisional": True,
                "detail": f"OpenStreetMap could not be reached, so {FALLBACK_FREEWAY_MPH:.0f} mph (the usual "
                          "California freeway limit) is assumed. It is not this road's posted sign; it "
                          "will be looked up again shortly."}
    return None


def lookup_speed_limit(lat: float, lon: float, route: str = "", timeout: float = 10.0) -> dict | None:
    """Speed limit for the road at (lat, lon): {"limit_mph", "source": osm|assumed|none, "detail"}.

    Answers found on OpenStreetMap are remembered in memory and on disk, so each camera needs only
    one successful lookup ever. When OpenStreetMap cannot be reached the result is a provisional
    fallback for Interstate/US routes (flagged "provisional": True, not remembered) or None."""
    key = (round(float(lat), 4), round(float(lon), 4), route)
    disk_key = f"{key[0]},{key[1]},{route}"
    if key in _cache:
        return _cache[key]
    on_disk = _load_disk_cache().get(disk_key)
    if on_disk:
        _cache[key] = on_disk
        return on_disk
    ways = _query_overpass(float(lat), float(lon), timeout)
    if ways is None:
        return fallback_for_route(route)  # provisional: not cached, so the next call tries again
    way = pick_road(ways, float(lat), float(lon), route)
    result = limit_from_way(way) if way else {"limit_mph": None, "source": "none",
                                              "detail": "No road found near this camera in OpenStreetMap."}
    _cache[key] = result
    _save_disk_cache(disk_key, result)
    return result


# ---------------------------------------------------------------- non-blocking use from a UI
_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="speed-limit")
_pending: dict[tuple, Future] = {}
_retry_after: dict[tuple, float] = {}
RETRY_SECONDS = 120.0


def lookup_async(lat: float, lon: float, route: str = "") -> dict:
    """Never blocks. Returns {"state": "done"|"pending", "result": dict | None}.

    "pending": the lookup is running in a background thread; `result` is the provisional fallback
    for the route (or None) to show meanwhile. Poll again later. A failed lookup is not retried for
    RETRY_SECONDS, so an unreachable service cannot keep spawning requests."""
    key = (round(float(lat), 4), round(float(lon), 4), route)
    if key in _cache:
        return {"state": "done", "result": _cache[key]}
    fut = _pending.get(key)
    if fut is not None and fut.done():
        del _pending[key]
        try:
            result = fut.result()
        except Exception:
            result = None
        if result is None or result.get("provisional"):
            _retry_after[key] = time.time() + RETRY_SECONDS
        return {"state": "done", "result": result}
    if fut is None:
        if time.time() < _retry_after.get(key, 0.0):
            return {"state": "done", "result": fallback_for_route(route)}
        _pending[key] = _executor.submit(lookup_speed_limit, float(lat), float(lon), route)
    return {"state": "pending", "result": fallback_for_route(route)}
