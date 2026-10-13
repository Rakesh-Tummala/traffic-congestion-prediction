"""Helpers for drawing a camera-based route on a map: order the chosen cameras
along their corridor, geocode place names (Nominatim, OpenStreetMap), fetch the
real road geometry between two points from OSRM (OpenStreetMap routing), find
the cameras that sit along it, and color each stretch of road by the nearest
camera's congestion reading. All of these services are free and need no API key.

The colors are estimates from sparse camera snapshots, not continuous traffic
data like a navigation app has. A stretch within MAX_CAMERA_DISTANCE_M of a
camera is colored from it with confidence; one up to FAR_CAMERA_DISTANCE_M away
is colored too but flagged as a weaker guess; beyond that it is left uncolored
(label None) instead of guessed."""
import math

import numpy as np
import requests

OSRM_URL = "https://router.project-osrm.org/route/v1/driving/{coords}"
NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"
# Nominatim's usage policy requires an identifying User-Agent and at most one request per second.
USER_AGENT = ("traffic-congestion-prediction/1.0 "
              "(student project; github.com/Rakesh-Tummala/traffic-congestion-prediction)")
EARTH_RADIUS_M = 6371000.0
MAX_CAMERA_DISTANCE_M = 2000.0   # confident coloring
FAR_CAMERA_DISTANCE_M = 8000.0   # weaker, faded coloring
CAMERA_ON_ROUTE_M = 1500.0       # a camera this close to the road is almost surely on it


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


def geocode(query: str, limit: int = 5, timeout: float = 10.0) -> list[dict]:
    """Look a place name up via Nominatim. Returns [{"name", "lat", "lon"}, ...],
    best match first; an empty list for a blank query or no match."""
    if not query.strip():
        return []
    resp = requests.get(NOMINATIM_URL, params={"q": query, "format": "jsonv2", "limit": limit},
                        headers={"User-Agent": USER_AGENT}, timeout=timeout)
    resp.raise_for_status()
    return [{"name": r["display_name"], "lat": float(r["lat"]), "lon": float(r["lon"])} for r in resp.json()]


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


def cameras_near_route(route_points: list[tuple[float, float]], cameras: list[dict],
                       max_distance_m: float = CAMERA_ON_ROUTE_M, max_cameras: int = 16) -> list[dict]:
    """Cameras within max_distance_m of the route, in the order they are met
    along it. If more than max_cameras qualify, keeps max_cameras evenly spaced
    ones so a long route stays quick to check. Cameras without usable
    coordinates are skipped."""
    if not route_points or not cameras:
        return []
    route = np.radians(np.array(route_points, dtype=float))  # (n, 2) lat, lon in radians

    usable, coords = [], []
    for cam in cameras:
        try:
            coords.append((float(cam["latitude"]), float(cam["longitude"])))
            usable.append(cam)
        except (KeyError, TypeError, ValueError):
            continue
    if not usable:
        return []
    cam_deg = np.array(coords, dtype=float)

    # Cheap bounding-box prefilter so the distance matrix below stays small.
    lat_margin = max_distance_m / 111_000.0 * 1.5
    lat_lo, lat_hi = np.degrees(route[:, 0]).min() - lat_margin, np.degrees(route[:, 0]).max() + lat_margin
    lon_margin = lat_margin / max(math.cos(math.radians((lat_lo + lat_hi) / 2)), 0.2)
    lon_lo, lon_hi = np.degrees(route[:, 1]).min() - lon_margin, np.degrees(route[:, 1]).max() + lon_margin
    keep = ((cam_deg[:, 0] >= lat_lo) & (cam_deg[:, 0] <= lat_hi)
            & (cam_deg[:, 1] >= lon_lo) & (cam_deg[:, 1] <= lon_hi))
    if not keep.any():
        return []
    cands = [usable[i] for i in np.flatnonzero(keep)]
    cam_rad = np.radians(cam_deg[keep])

    dlat = cam_rad[:, None, 0] - route[None, :, 0]
    dlon = cam_rad[:, None, 1] - route[None, :, 1]
    a = (np.sin(dlat / 2) ** 2
         + np.cos(cam_rad[:, None, 0]) * np.cos(route[None, :, 0]) * np.sin(dlon / 2) ** 2)
    dist = 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # (cameras, route points)
    nearest_idx = dist.argmin(axis=1)
    nearest_dist = dist.min(axis=1)

    hits = [(int(nearest_idx[k]), cands[k]) for k in range(len(cands)) if nearest_dist[k] <= max_distance_m]
    hits.sort(key=lambda h: h[0])
    ordered = [cam for _, cam in hits]
    if len(ordered) > max_cameras:
        picks = sorted({int(round(i)) for i in np.linspace(0, len(ordered) - 1, max_cameras)})
        ordered = [ordered[i] for i in picks]
    return ordered


def color_route(route_points: list[tuple[float, float]], readings: list[tuple[float, float, str]],
                max_distance_m: float = MAX_CAMERA_DISTANCE_M,
                far_distance_m: float = FAR_CAMERA_DISTANCE_M) -> list[dict]:
    """Split a route into runs that share a label and confidence: each point takes
    the label of the nearest camera reading (lat, lon, label). Within
    max_distance_m that is a confident ("near") label; up to far_distance_m it is
    kept but marked near=False (a weaker guess); beyond that the label is None.
    Consecutive runs share their boundary point so the drawn line has no gaps.
    Returns [{"label": str | None, "near": bool, "path": [[lon, lat], ...]}]."""
    def label_at(lat: float, lon: float):
        best_d, best_label = float("inf"), None
        for r_lat, r_lon, label in readings:
            d = haversine_m(lat, lon, r_lat, r_lon)
            if d < best_d:
                best_d, best_label = d, label
        if best_d <= max_distance_m:
            return best_label, True
        if best_d <= far_distance_m:
            return best_label, False
        return None, False

    runs: list[dict] = []
    for lat, lon in route_points:
        label, near = label_at(lat, lon)
        point = [lon, lat]
        if runs and runs[-1]["label"] == label and runs[-1]["near"] == near:
            runs[-1]["path"].append(point)
        else:
            boundary = [runs[-1]["path"][-1]] if runs else []
            runs.append({"label": label, "near": near, "path": boundary + [point]})
    return runs


def fit_view(points: list, width_px: int = 700, height_px: int = 420, padding: float = 0.88) -> tuple[float, float, float]:
    """(latitude, longitude, zoom) that fits every [lon, lat] point inside a map of
    the given pixel size, using the Web Mercator maths deck.gl uses (512px base
    tile). pydeck's own compute_view assumes a squarer box, so a tall north-south
    route in a short, wide map got its ends cropped (found via live testing).
    width_px is a conservative guess for a narrow window; wider maps just get margin."""
    lons = [p[0] for p in points]
    lats = [p[1] for p in points]

    def merc_y(lat_deg: float) -> float:
        return math.log(math.tan(math.pi / 4 + math.radians(lat_deg) / 2))

    lon_span = max(max(lons) - min(lons), 1e-4)
    y_span = max(merc_y(max(lats)) - merc_y(min(lats)), 1e-6)
    zoom_x = math.log2(width_px * 360 / (512 * lon_span))
    zoom_y = math.log2(height_px * 2 * math.pi / (512 * y_span))
    zoom = max(1.0, min(zoom_x, zoom_y, 15.0) + math.log2(padding))
    mid_y = (merc_y(max(lats)) + merc_y(min(lats))) / 2
    mid_lat = math.degrees(2 * math.atan(math.exp(mid_y)) - math.pi / 2)
    return mid_lat, (max(lons) + min(lons)) / 2, zoom
