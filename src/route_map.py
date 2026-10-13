"""Helpers for drawing a camera-based route on a map: order the chosen cameras
along their corridor, fetch the real road geometry between its two ends from
OSRM (OpenStreetMap routing, free, no API key), and color each stretch of that
road by the nearest camera's congestion reading.

The colors are estimates from sparse camera snapshots, not continuous traffic
data like a navigation app has: a stretch with no camera within
MAX_CAMERA_DISTANCE_M is left uncolored (label None) instead of guessed."""
import math

import requests

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{coords}"
EARTH_RADIUS_M = 6371000.0
MAX_CAMERA_DISTANCE_M = 2000.0


def haversine_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * EARTH_RADIUS_M * math.asin(math.sqrt(a))


def order_cameras(cameras: list[dict]) -> list[dict]:
    """Order cameras from one end of their corridor to the other: start at the
    western end of the farthest-apart pair, then repeatedly hop to the nearest
    unvisited camera. A rough path for cameras along one road, not a solved
    routing problem. Each camera needs float-able "latitude" and "longitude"."""
    pts = [(float(c["latitude"]), float(c["longitude"])) for c in cameras]
    n = len(cameras)
    if n < 3:
        return sorted(cameras, key=lambda c: (float(c["longitude"]), float(c["latitude"])))

    best, start = -1.0, 0
    for i in range(n):
        for j in range(i + 1, n):
            d = haversine_m(*pts[i], *pts[j])
            if d > best:
                best = d
                start = i if (pts[i][1], pts[i][0]) <= (pts[j][1], pts[j][0]) else j

    order, remaining = [start], set(range(n)) - {start}
    while remaining:
        last = order[-1]
        nxt = min(remaining, key=lambda k: haversine_m(*pts[last], *pts[k]))
        order.append(nxt)
        remaining.remove(nxt)
    return [cameras[i] for i in order]


def fetch_route(start: tuple[float, float], end: tuple[float, float], timeout: float = 10.0) -> dict:
    """Driving route between two (lat, lon) points. Returns {"points": [(lat, lon), ...],
    "distance_m", "duration_s"}; duration is free-flow time and excludes traffic.
    Raises on a network error or when OSRM finds no route."""
    coords = f"{start[1]},{start[0]};{end[1]},{end[0]}"
    resp = requests.get(OSRM_URL.format(coords=coords),
                        params={"overview": "full", "geometries": "geojson"}, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if data.get("code") != "Ok" or not data.get("routes"):
        raise ValueError(f"No driving route found ({data.get('code', 'unknown')})")
    route = data["routes"][0]
    return {
        "points": [(lat, lon) for lon, lat in route["geometry"]["coordinates"]],
        "distance_m": float(route["distance"]),
        "duration_s": float(route["duration"]),
    }


def color_route(route_points: list[tuple[float, float]], readings: list[tuple[float, float, str]],
                max_distance_m: float = MAX_CAMERA_DISTANCE_M) -> list[dict]:
    """Split a route into runs that share a label: each point takes the label of
    the nearest camera reading (lat, lon, label) if one is within max_distance_m,
    otherwise None. Consecutive runs share their boundary point so the drawn line
    has no gaps. Returns [{"label": str | None, "path": [[lon, lat], ...]}]."""
    def label_at(lat: float, lon: float):
        best_d, best_label = float("inf"), None
        for r_lat, r_lon, label in readings:
            d = haversine_m(lat, lon, r_lat, r_lon)
            if d < best_d:
                best_d, best_label = d, label
        return best_label if best_d <= max_distance_m else None

    runs: list[dict] = []
    for lat, lon in route_points:
        label = label_at(lat, lon)
        point = [lon, lat]
        if runs and runs[-1]["label"] == label:
            runs[-1]["path"].append(point)
        else:
            boundary = [runs[-1]["path"][-1]] if runs else []
            runs.append({"label": label, "path": boundary + [point]})
    return runs
