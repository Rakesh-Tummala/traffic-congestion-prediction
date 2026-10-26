"""Evaluate vehicle counting on a hand-counted, time-stratified frame set, for one or more detector weights.

Inputs: research/data/label_set.csv (from sample_for_labels.py) and a filled-in labels CSV
(id, count, usable, note). Reports, per weights file and per time-of-day bucket:
count MAE, bias, within-tolerance rate, and the correlation of detected and true counts,
with bootstrap 95% intervals over frames, and a paired comparison between two weights files.

    python research/detector_eval.py --weights yolov8s.pt yolov8m.pt runs/finetune/weights/best.pt --conf 0.08
Run from the project root (weights paths are relative to it).
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

DATA = os.path.join(HERE, "data")
RESULTS = os.path.join(HERE, "results")


def detect_counts(weights, files, conf, imgsz=640):
    import cv2
    from ultralytics import YOLO
    from cv_module import VEHICLE_CLASS_IDS
    model = YOLO(weights)
    out = []
    for f in files:
        img = cv2.imread(f)
        r = model.predict(img, conf=conf, imgsz=imgsz, classes=list(VEHICLE_CLASS_IDS), verbose=False)[0]
        out.append(len(r.boxes))
    return np.array(out)


def summarize(err, true, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    n = len(err)
    idx = rng.integers(0, n, size=(n_boot, n))
    mae_b = np.abs(err)[idx].mean(axis=1)
    lo, hi = np.percentile(mae_b, [2.5, 97.5])
    tol = np.maximum(2, 0.25 * true)
    return {"n": n, "MAE": float(np.abs(err).mean()), "MAE_lo": float(lo), "MAE_hi": float(hi), "bias": float(err.mean()),
            "within_tol": float((np.abs(err) <= tol).mean())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--labels", default=os.path.join(DATA, "labels_new.csv"))
    ap.add_argument("--weights", nargs="+", default=["yolov8s.pt"])
    ap.add_argument("--conf", type=float, default=0.08)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()
    meta = pd.read_csv(os.path.join(DATA, "label_set.csv"))
    lab = pd.read_csv(args.labels)
    df = meta.merge(lab, on="id")
    df = df[(df["usable"].astype(str).str.lower() == "yes") & df["count"].notna()].copy()
    df["count"] = df["count"].astype(float)
    files = [os.path.join(DATA, f) for f in df["file"]]
    print(f"{len(df)} labelled usable frames")
    rows, errs = [], {}
    for w in args.weights:
        pred = detect_counts(w, files, args.conf, args.imgsz)
        df[f"pred_{w}"] = pred
        err = pred - df["count"].to_numpy()
        errs[w] = err
        rows.append({"weights": w, "bucket": "ALL", **summarize(err, df["count"].to_numpy()),
                     "r": float(np.corrcoef(pred, df["count"])[0, 1]) if len(df) > 2 else np.nan})
        for b, g in df.groupby("bucket"):
            e = df.loc[g.index, f"pred_{w}"].to_numpy() - g["count"].to_numpy()
            rows.append({"weights": w, "bucket": b, **summarize(e, g["count"].to_numpy()), "r": np.nan})
    res = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    res.to_csv(os.path.join(RESULTS, "detector_by_time.csv"), index=False)
    print(res.to_string(index=False, float_format=lambda v: f"{v:.2f}"))
    if len(args.weights) >= 2:
        rng = np.random.default_rng(1)
        base = np.abs(errs[args.weights[0]])
        for w in args.weights[1:]:
            d = np.abs(errs[w]) - base
            idx = rng.integers(0, len(d), size=(2000, len(d)))
            lo, hi = np.percentile(d[idx].mean(axis=1), [2.5, 97.5])
            print(f"paired |error| difference {w} - {args.weights[0]}: {d.mean():+.2f} (95% CI {lo:+.2f} to {hi:+.2f})")


if __name__ == "__main__":
    main()
