"""Readers and helpers for Caltrans PeMS loop-detector data (the ground truth for camera validation).

PeMS (https://pems.dot.ca.gov) needs a free account; the files are downloaded from its Data Clearinghouse,
so this repository cannot fetch them. The formats below follow PeMS's published descriptions:

* Station 5-minute files (``d07_text_station_5min_YYYY_MM_DD.txt[.gz]``): comma-separated, NO header row.
  Columns: 1 timestamp (MM/DD/YYYY HH:MM:SS, Pacific local time), 2 station id, 3 district, 4 freeway,
  5 direction (N/S/E/W), 6 lane type (ML = mainline), 7 station length, 8 samples, 9 % observed,
  10 total flow (vehicles per 5 min, all lanes), 11 average occupancy (0-1), 12 average speed (mph),
  then five columns per lane that are not used here.
* Station metadata (``d07_text_meta_YYYY_MM_DD.txt``): tab-separated WITH a header row that includes
  ID, Fwy, Dir, District, County, City, State_PM, Abs_PM, Latitude, Longitude, Length, Type, Lanes, Name.

These parsers were written from that description and checked against synthetic files in the same layout
(tests/test_pems.py). Run them on the first real download and look at ``describe()`` before trusting results.
"""
import gzip
import io
import math
import re

import numpy as np
import pandas as pd

STATION_COLUMNS = ["ts", "station", "district", "freeway", "direction", "lane_type", "length", "samples",
                   "pct_observed", "flow", "occupancy", "speed"]


def _open_text(path):
    if str(path).endswith(".gz"):
        return io.TextIOWrapper(gzip.open(path, "rb"), encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def parse_station_5min(path) -> pd.DataFrame:
    with _open_text(path) as f:
        df = pd.read_csv(f, header=None, usecols=range(12), names=STATION_COLUMNS, low_memory=False)
    df["ts"] = pd.to_datetime(df["ts"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    for c in ("flow", "occupancy", "speed", "pct_observed", "length"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["ts"]).reset_index(drop=True)


def parse_meta(path) -> pd.DataFrame:
    with _open_text(path) as f:
        df = pd.read_csv(f, sep="\t", low_memory=False)
    df = df.rename(columns={"ID": "station", "Fwy": "freeway", "Dir": "direction", "Latitude": "lat",
                            "Longitude": "lon", "Type": "type", "Lanes": "lanes", "Abs_PM": "abs_pm", "Name": "name"})
    for c in ("lat", "lon"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna(subset=["lat", "lon"]).reset_index(drop=True)


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    r = 6371.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi, dlmb = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlmb / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def route_number(route: str) -> int | None:
    """'I-5' -> 5, 'US-101' -> 101, 'SR-134' -> 134, '' -> None."""
    m = re.search(r"(\d+)", route or "")
    return int(m.group(1)) if m else None


def match_stations(cam_lat, cam_lon, cam_route, meta: pd.DataFrame, max_km: float = 1.5) -> list[dict]:
    """Nearest mainline station per direction of travel on the camera's freeway, within `max_km`."""
    fw = route_number(cam_route)
    cand = meta[(meta["freeway"] == fw)]
    if "type" in cand.columns:
        cand = cand[cand["type"].astype(str).str.upper() == "ML"]
    out = {}
    for _, r in cand.iterrows():
        d = haversine_km(cam_lat, cam_lon, r["lat"], r["lon"])
        if d <= max_km and (r["direction"] not in out or d < out[r["direction"]]["km"]):
            out[r["direction"]] = {"station": int(r["station"]), "direction": r["direction"], "km": round(d, 2)}
    return list(out.values())


def combine_directions(stations: pd.DataFrame, chosen: list[dict]) -> pd.DataFrame:
    """5-minute series for a camera: flows add up, occupancy and speed are flow-weighted over directions."""
    ids = [c["station"] for c in chosen]
    sub = stations[stations["station"].isin(ids)].copy()
    if sub.empty:
        return pd.DataFrame(columns=["ts", "flow", "occupancy", "speed"])
    sub["w"] = sub["flow"].fillna(0).clip(lower=0) + 1e-6
    g = sub.groupby("ts")
    out = pd.DataFrame({
        "flow": g["flow"].sum(min_count=1),
        "occupancy": g.apply(lambda d: np.average(d["occupancy"].fillna(0), weights=d["w"]), include_groups=False),
        "speed": g.apply(lambda d: np.average(d["speed"].fillna(d["speed"].mean()), weights=d["w"]), include_groups=False)
        if sub["speed"].notna().any() else np.nan,
    }).reset_index()
    return out


# Congestion levels from PeMS speed (mph). A documented choice, varied in the sensitivity analysis.
def speed_level(speed: float, edges=(55.0, 45.0, 30.0)) -> int:
    """0 Low, 1 Moderate, 2 High, 3 Severe from the average speed of the 5-minute bin."""
    if not np.isfinite(speed):
        return -1
    return 0 if speed >= edges[0] else 1 if speed >= edges[1] else 2 if speed >= edges[2] else 3


LEVEL_NAMES = ["Low", "Moderate", "High", "Severe"]
