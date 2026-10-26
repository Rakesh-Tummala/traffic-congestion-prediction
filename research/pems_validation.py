"""Validate the camera pipeline against PeMS loop-detector ground truth.

Inputs (see research/pems.py for the formats):
  --camera-log   research/data/camera_log.csv, written by research/log_cameras.py (5-minute bins)
  --meta         a PeMS station metadata file for the district
  --stations     one or more PeMS 5-minute station files covering the same days as the camera log

Studies (each writes a CSV and a section of research/results/pems_validation.md):
  A3  camera vehicle count / image density against detector flow and occupancy, with a time-offset scan
      and a held-out (later in time) calibration of count -> flow.
  A4  the tracker's raw median speed against the detector speed: raw error, the app's "scale to a typical
      speed" rule, and a per-camera zoom scale fitted on the earlier half of the days.
  A6  congestion-level agreement with ground truth (speed-based) for: camera only, history only,
      the "more severe of the two" rule, the live-first rule used now, and a small learned fusion.

    python research/pems_validation.py --camera-log research/data/camera_log.csv \
        --meta d07_text_meta_2025_01_01.txt --stations d07_text_station_5min_2025_01_*.txt.gz
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, mean_absolute_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import pems  # noqa: E402

RESULTS = os.path.join(HERE, "results")
LEVELS = ["Low", "Moderate", "High", "Severe"]


def load_joined(camera_log, meta_path, station_paths, max_km, offset_min=0):
    cam = pd.read_csv(camera_log)
    cam = cam[cam["status"] == "ok"].copy()
    cam["bin"] = pd.to_datetime(cam["pacific"]).dt.floor("5min") + pd.Timedelta(minutes=offset_min)
    meta = pems.parse_meta(meta_path)
    stations = pd.concat([pems.parse_station_5min(p) for p in station_paths], ignore_index=True)
    stations = stations[stations["lane_type"].astype(str).str.upper() == "ML"]
    frames, matches = [], []
    for cid, g in cam.groupby("camera_id"):
        first = g.iloc[0]
        chosen = pems.match_stations(first["latitude"], first["longitude"], first["route"], meta, max_km)
        matches.append({"camera_id": cid, "name": first["name"], "stations": chosen})
        if not chosen:
            continue
        series = pems.combine_directions(stations, chosen).rename(columns={"ts": "bin"})
        j = g.merge(series, on="bin", how="inner")
        j["station_km"] = max(c["km"] for c in chosen)
        frames.append(j)
    joined = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    return joined, matches


# ------------------------------------------------------------------ A3
def study_a3(cam_log, meta_path, station_paths, max_km):
    rows = []
    for off in range(-10, 11, 5):
        j, _ = load_joined(cam_log, meta_path, station_paths, max_km, off)
        if j.empty:
            continue
        for x in ("vehicle_count", "density"):
            for y in ("flow", "occupancy"):
                d = j[[x, y]].dropna()
                if len(d) > 10:
                    rows.append({"offset_min": off, "camera": x, "detector": y, "n": len(d),
                                 "pearson": d[x].corr(d[y]), "spearman": d[x].corr(d[y], method="spearman")})
    scan = pd.DataFrame(rows)
    best = int(scan.loc[scan[(scan.camera == "vehicle_count") & (scan.detector == "flow")]["pearson"].idxmax(), "offset_min"]) \
        if not scan.empty else 0
    j, matches = load_joined(cam_log, meta_path, station_paths, max_km, best)
    per_cam, held = [], []
    for cid, g in j.groupby("camera_id"):
        g = g.sort_values("bin").dropna(subset=["vehicle_count", "flow"])
        if len(g) < 30:
            continue
        cut = int(len(g) * 0.6)
        tr, te = g.iloc[:cut], g.iloc[cut:]
        slope = np.polyfit(tr["vehicle_count"], tr["flow"], 1)
        pred = np.polyval(slope, te["vehicle_count"])
        per_cam.append({"camera": g["name"].iloc[0], "n": len(g), "r_count_flow": g["vehicle_count"].corr(g["flow"]),
                        "r_density_occ": g["density"].corr(g["occupancy"]),
                        "heldout_R2_flow": r2_score(te["flow"], pred), "heldout_MAE_flow": mean_absolute_error(te["flow"], pred),
                        "station_km": g["station_km"].iloc[0]})
    return scan, best, pd.DataFrame(per_cam), j, matches


# ------------------------------------------------------------------ A6
def level_metrics(y, p):
    y, p = np.asarray(y), np.asarray(p)
    return {"n": len(y), "accuracy": float(np.mean(y == p)), "within1": float(np.mean(np.abs(y - p) <= 1)),
            "balanced_acc": float(balanced_accuracy_score(y, p)), "macro_F1": float(f1_score(y, p, average="macro")),
            "mean_abs_level_error": float(np.mean(np.abs(y - p)))}


def study_a6(j, speed_edges=(55.0, 45.0, 30.0), use_minnesota=True):
    d = j.dropna(subset=["speed", "density"]).copy()
    d["gt"] = d["speed"].map(lambda s: pems.speed_level(s, speed_edges))
    d = d[d["gt"] >= 0].sort_values("bin").reset_index(drop=True)
    if len(d) < 60:
        return pd.DataFrame(), d
    from cv_module import congestion_from_density
    d["cam"] = d["density"].map(lambda v: LEVELS.index(congestion_from_density(v)))
    cut = int(len(d) * 0.6)
    tr, te = d.iloc[:cut], d.iloc[cut:].copy()
    # history from the earlier days only: mean speed by (hour, weekday) -> level
    prof = tr.assign(h=tr["bin"].dt.hour, w=tr["bin"].dt.dayofweek).groupby(["h", "w"])["speed"].mean()
    fallback = tr["speed"].mean()
    te["hist"] = [pems.speed_level(prof.get((b.hour, b.dayofweek), fallback), speed_edges) for b in te["bin"]]
    te["hist"] = te["hist"].clip(lower=0)
    # the Minnesota-trained app model, if its files are present
    try:
        if not use_minnesota:
            raise RuntimeError('skipped')
        from datetime import datetime
        from live_predict import calibrated_label, predict_all_models
        te["hist_mn"] = [LEVELS.index(calibrated_label(
            predict_all_models(datetime(b.year, b.month, b.day, b.hour, b.minute), 15.0, 0.0, 0.0, 20, 0, "Clear")["lstm"]))
            for b in te["bin"]]
    except Exception:
        te["hist_mn"] = np.nan
    rules = {
        "Camera only (density)": te["cam"],
        "History only (PeMS hour x weekday)": te["hist"],
        "More severe of camera and PeMS history": np.maximum(te["cam"], te["hist"]),
    }
    if te["hist_mn"].notna().all():
        rules["History only (Minnesota model)"] = te["hist_mn"].astype(int)
        rules["More severe of camera and Minnesota model"] = np.maximum(te["cam"], te["hist_mn"].astype(int))
    # learned fusion: logistic regression on density, count and the history level (fit on the earlier 60%)
    tr = tr.copy()
    tr["hist"] = [pems.speed_level(prof.get((b.hour, b.dayofweek), fallback), speed_edges) for b in tr["bin"]]
    feats = lambda x: np.c_[np.log1p(x["density"] * 100), x["vehicle_count"], x["hist"].clip(lower=0), np.sin(2 * np.pi * x["bin"].dt.hour / 24), np.cos(2 * np.pi * x["bin"].dt.hour / 24)]  # noqa: E731
    if tr["gt"].nunique() > 1:
        clf = LogisticRegression(max_iter=2000).fit(feats(tr), tr["gt"])
        rules["Learned fusion (logistic regression)"] = clf.predict(feats(te))
    rows = [{"rule": k, **level_metrics(te["gt"], v)} for k, v in rules.items()]
    return pd.DataFrame(rows), te


# ------------------------------------------------------------------ A4
def study_a4(j):
    d = j.dropna(subset=["raw_speed_median_mph", "speed"]).copy()
    d = d[(d["raw_speed_median_mph"] > 3) & (d["burst_tracks"] >= 6)].sort_values("bin")
    if len(d) < 8:
        return pd.DataFrame(), pd.DataFrame()
    rows, per_cam = [], []
    for cid, g in d.groupby("camera_id"):
        if len(g) < 8:
            continue
        cut = len(g) // 2
        tr, te = g.iloc[:cut], g.iloc[cut:]
        k = float(np.median(tr["speed"] / tr["raw_speed_median_mph"]))        # per-camera zoom scale from the earlier half
        ref = 60.0                                                            # the app's default "typical traffic speed"
        mono = {"unscaled raw": te["raw_speed_median_mph"],
                "app rule: scale median to 60 mph": te["raw_speed_median_mph"] * (ref / te["raw_speed_median_mph"].median()),
                "per-camera scale fitted on earlier half": te["raw_speed_median_mph"] * k}
        for name, p in mono.items():
            rows.append({"camera": g["name"].iloc[0], "method": name, "n_test": len(te),
                         "MAE_mph": mean_absolute_error(te["speed"], p), "corr": float(np.corrcoef(te["speed"], p)[0, 1]) if len(te) > 2 else np.nan})
        per_cam.append({"camera": g["name"].iloc[0], "n": len(g), "median_scale_pems_over_raw": k,
                        "scale_IQR": float(np.subtract(*np.percentile(g["speed"] / g["raw_speed_median_mph"], [75, 25])))})
    res = pd.DataFrame(rows)
    summary = res.groupby("method").agg(MAE_mph=("MAE_mph", "mean"), corr=("corr", "mean"), cameras=("camera", "nunique")).reset_index()
    return summary, pd.DataFrame(per_cam)


def md(df):
    if df is None or len(df) == 0:
        return "_no data_"
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        out.append("| " + " | ".join(f"{v:.3f}" if isinstance(v, (float, np.floating)) else str(v) for v in r) + " |")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--camera-log", default=os.path.join(HERE, "data", "camera_log.csv"))
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stations", nargs="+", required=True, help="PeMS 5-minute station files (globs allowed)")
    ap.add_argument("--max-km", type=float, default=1.5)
    args = ap.parse_args()
    paths = sorted({p for pat in args.stations for p in glob.glob(pat)})
    if not paths:
        sys.exit("no station files matched")
    os.makedirs(RESULTS, exist_ok=True)

    scan, best, per_cam, joined, matches = study_a3(args.camera_log, args.meta, paths, args.max_km)
    scan.to_csv(f"{RESULTS}/pems_offset_scan.csv", index=False)
    per_cam.to_csv(f"{RESULTS}/pems_a3_per_camera.csv", index=False)
    a6, test_rows = study_a6(joined)
    a6.to_csv(f"{RESULTS}/pems_a6_fusion.csv", index=False)
    a4_sum, a4_cam = study_a4(joined)
    a4_sum.to_csv(f"{RESULTS}/pems_a4_speed.csv", index=False)
    a4_cam.to_csv(f"{RESULTS}/pems_a4_per_camera.csv", index=False)

    lines = ["# Camera vs PeMS ground truth", "",
             f"Joined 5-minute rows: {len(joined)}; cameras matched to a station: "
             f"{sum(1 for m in matches if m['stations'])}/{len(matches)}.", "",
             "## A3. Count and density against detector flow and occupancy (time-offset scan)", "", md(scan),
             f"\nBest alignment offset: {best} min (the correlations below use it).", "",
             "### Per camera, with count->flow calibrated on the first 60% of the time and tested on the rest", "", md(per_cam), "",
             "## A4. Tracker speed against detector speed", "", md(a4_sum), "", md(a4_cam), "",
             "## A6. Congestion-level agreement with ground truth (speed levels 55/45/30 mph)", "", md(a6), ""]
    open(f"{RESULTS}/pems_validation.md", "w", encoding="utf-8").write("\n".join(lines))
    print("wrote", f"{RESULTS}/pems_validation.md")


if __name__ == "__main__":
    main()
