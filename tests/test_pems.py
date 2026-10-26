"""Tests for the PeMS readers and the camera-vs-detector validation, on synthetic files in PeMS's layout.

These prove the code paths work and recover a known relationship; they do NOT prove the parsers match a real
PeMS download (check the first real file by eye)."""
import gzip
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, os.path.join(HERE, "..", "research"))

import pems  # noqa: E402
import pems_validation as pv  # noqa: E402

CAM_LAT, CAM_LON = 34.17, -118.30


def make_files(tmp_path, days=5):
    meta = pd.DataFrame({
        "ID": [1001, 1002, 2001, 3001], "Fwy": [5, 5, 5, 101], "Dir": ["N", "S", "N", "N"], "District": 7,
        "County": 37, "City": 0, "State_PM": "1", "Abs_PM": [1.0, 1.0, 40.0, 1.0],
        "Latitude": [CAM_LAT + 0.002, CAM_LAT - 0.002, CAM_LAT + 0.5, CAM_LAT + 0.001],
        "Longitude": [CAM_LON] * 4, "Length": 1.0, "Type": ["ML", "ML", "ML", "ML"], "Lanes": 4, "Name": "x"})
    meta_path = tmp_path / "meta.txt"
    meta.to_csv(meta_path, sep="\t", index=False)

    rng = np.random.default_rng(1)
    idx = pd.date_range("2025-03-03", periods=days * 288, freq="5min")
    hour = np.asarray(idx.hour + idx.minute / 60, dtype=float)
    demand = 0.5 + 0.5 * np.exp(-((hour - 8) ** 2) / 4) + 0.6 * np.exp(-((hour - 17) ** 2) / 6)   # two rush hours
    rows = []
    for sid, share in ((1001, 1.0), (1002, 0.8)):
        flow = (demand * 420 * share + rng.normal(0, 8, len(idx))).clip(5)
        speed = (68 - 38 * (demand / demand.max()) ** 2 + rng.normal(0, 1.5, len(idx))).clip(8)
        occ = (flow / 1500 + rng.normal(0, 0.004, len(idx))).clip(0.001)
        for t, f, s, o in zip(idx, flow, speed, occ):
            rows.append([t.strftime("%m/%d/%Y %H:%M:%S"), sid, 7, 5, "N" if sid == 1001 else "S", "ML", 0.5, 20, 100,
                         round(f), round(o, 4), round(s, 1)] + [0] * 10)
    st_path = tmp_path / "d07_text_station_5min_2025_03_03.txt.gz"
    with gzip.open(st_path, "wt") as f:
        pd.DataFrame(rows).to_csv(f, header=False, index=False)

    # camera: counts proportional to total flow, raw speed = true speed / 1.35 (the "wrong zoom")
    total_flow = pd.Series(0.0, index=idx)
    for sid, share in ((1001, 1.0), (1002, 0.8)):
        total_flow += pd.Series(demand * 420 * share, index=idx)
    cam_rows = []
    avg_speed = 68 - 38 * (demand / demand.max()) ** 2
    for i, t in enumerate(idx):
        count = max(0, round(total_flow.iloc[i] / 14 + rng.normal(0, 1.2)))
        density = min(0.9, count * 0.006 + abs(rng.normal(0, 0.003)))
        cam_rows.append({"utc": "", "pacific": t.strftime("%Y-%m-%d %H:%M"), "camera_id": "cam1", "name": "I-5 : test",
                         "route": "I-5", "latitude": CAM_LAT, "longitude": CAM_LON, "status": "ok", "vehicle_count": count,
                         "density": density, "level": "Low", "burst_tracks": 14 if i % 3 == 0 else np.nan,
                         "raw_speed_median_mph": (avg_speed[i] / 1.35 + rng.normal(0, 1.0)) if i % 3 == 0 else np.nan,
                         "scale": np.nan, "camera_height_m": 8.0, "fit_corr": 0.9})
    cam_path = tmp_path / "camera_log.csv"
    pd.DataFrame(cam_rows).to_csv(cam_path, index=False)
    return str(cam_path), str(meta_path), [str(st_path)]


def test_parsers_read_the_documented_layout(tmp_path):
    cam, meta, stations = make_files(tmp_path)
    m = pems.parse_meta(meta)
    assert set(["station", "freeway", "direction", "lat", "lon"]).issubset(m.columns)
    s = pems.parse_station_5min(stations[0])
    assert list(s.columns) == pems.STATION_COLUMNS
    assert s["ts"].dtype.kind == "M" and s["flow"].notna().all()


def test_station_matching_picks_nearest_mainline_station_per_direction_on_the_right_freeway(tmp_path):
    _, meta, _ = make_files(tmp_path)
    m = pems.parse_meta(meta)
    chosen = pems.match_stations(CAM_LAT, CAM_LON, "I-5", m, max_km=1.5)
    assert {c["station"] for c in chosen} == {1001, 1002}        # not the far one, not US-101
    assert pems.match_stations(CAM_LAT, CAM_LON, "I-405", m) == []


def test_a3_recovers_the_count_flow_relationship(tmp_path):
    cam, meta, stations = make_files(tmp_path)
    scan, best, per_cam, joined, matches = pv.study_a3(cam, meta, stations, 1.5)
    assert best == 0
    top = scan[(scan.offset_min == 0) & (scan.camera == "vehicle_count") & (scan.detector == "flow")].iloc[0]
    assert top["pearson"] > 0.95
    assert per_cam["heldout_R2_flow"].iloc[0] > 0.9


def test_a4_recovers_the_hidden_zoom_scale(tmp_path):
    cam, meta, stations = make_files(tmp_path)
    *_, joined, _ = pv.study_a3(cam, meta, stations, 1.5)
    summary, per_cam = pv.study_a4(joined)
    assert abs(per_cam["median_scale_pems_over_raw"].iloc[0] - 1.35) < 0.08
    fitted = summary.set_index("method").loc["per-camera scale fitted on earlier half", "MAE_mph"]
    raw = summary.set_index("method").loc["unscaled raw", "MAE_mph"]
    assert fitted < raw / 2


def test_a6_compares_rules_and_a_learned_fusion_adapts_to_the_ground_truth(tmp_path):
    cam, meta, stations = make_files(tmp_path)
    *_, joined, _ = pv.study_a3(cam, meta, stations, 1.5)
    table, _ = pv.study_a6(joined, use_minnesota=False)
    t = table.set_index("rule")
    assert {"Camera only (density)", "History only (PeMS hour x weekday)", "Learned fusion (logistic regression)"} <= set(t.index)
    assert ((t[["accuracy", "within1", "balanced_acc", "macro_F1"]] >= 0) & (t[["accuracy", "within1", "balanced_acc", "macro_F1"]] <= 1)).all().all()
    # fixed density thresholds are arbitrary relative to speed levels; a fusion fitted on the earlier days is not
    assert t.loc["Learned fusion (logistic regression)", "accuracy"] >= t.loc["Camera only (density)", "accuracy"]


def test_speed_level_thresholds():
    assert [pems.speed_level(s) for s in (65, 50, 40, 20)] == [0, 1, 2, 3]
    assert pems.speed_level(float("nan")) == -1
