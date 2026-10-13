"""Evaluate the computer-vision part against hand-labelled Caltrans frames.

Three questions, answered on cv_eval/labels.csv (usable frames only):

1. Counting: how far is YOLO's vehicle count from the hand count? Swept over
   model size (n/s/m), input size and confidence threshold.
2. Congestion label: how often does the app's fixed density -> label rule
   (0.05 / 0.15 / 0.30) match the hand-labelled class?
3. Threshold fitting: if the cut-offs are fitted to labelled data instead, how
   much better is the label accuracy on frames the fit never saw? Measured
   with repeated k-fold cross-validation, so the number is out-of-sample.

Raw detections are cached in cv_eval/results/detections/, so changing the
confidence threshold or the metrics does not re-run the model.

    python cv_eval/evaluate_cv.py                 # full sweep
    python cv_eval/evaluate_cv.py --models yolov8s.pt --imgsz 640
"""
import argparse
import itertools
import json
import os
import sys

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(ROOT, "src"))

from cv_module import VEHICLE_CLASS_IDS, detect_vehicles_raw  # noqa: E402

LEVELS = ["Low", "Moderate", "High", "Severe"]
CURRENT_THRESHOLDS = (0.05, 0.15, 0.30)  # congestion_from_density in src/cv_module.py
DETECT_CONF = 0.03                       # detections are cached at this floor, filtered later
RESULTS = os.path.join(HERE, "results")
DET_DIR = os.path.join(RESULTS, "detections")


# ---------------------------------------------------------------- detection
def detect_all(weights: str, imgsz: int, ids: list[str], tiled: bool = False) -> dict:
    """{frame_id: [[conf, area_fraction], ...]} for every vehicle box above DETECT_CONF.
    tiled=True uses overlapping windows as well (cv_module.detect_vehicles_raw)."""
    os.makedirs(DET_DIR, exist_ok=True)
    stem = os.path.splitext(os.path.basename(weights))[0]
    cache = os.path.join(DET_DIR, f"{stem}_{imgsz}{'_tiled' if tiled else ''}_c{DETECT_CONF}.json")
    cached = json.load(open(cache)) if os.path.exists(cache) else {}
    todo = [i for i in ids if i not in cached]
    if todo:
        from ultralytics import YOLO
        model = YOLO(os.path.join(ROOT, weights))
        for n, fid in enumerate(todo, 1):
            frame = cv2.imread(os.path.join(HERE, "frames", f"{fid}.jpg"))
            h, w = frame.shape[:2]
            dets = detect_vehicles_raw(model, frame, DETECT_CONF, imgsz=imgsz, tiled=tiled)
            cached[fid] = [[d["conf"], max(0.0, d["box"][2] - d["box"][0]) * max(0.0, d["box"][3] - d["box"][1]) / (h * w)]
                           for d in dets]
            if n % 25 == 0:
                print(f"  {weights}@{imgsz}{' tiled' if tiled else ''}: {n}/{len(todo)}", flush=True)
        json.dump(cached, open(cache, "w"))
    return cached


def counts_and_density(dets: dict, ids: list[str], conf: float):
    count, density = [], []
    for fid in ids:
        kept = [b for b in dets[fid] if b[0] >= conf]
        count.append(len(kept))
        density.append(min(1.0, sum(b[1] for b in kept)))
    return np.array(count), np.array(density)


# ---------------------------------------------------------------- metrics
def count_metrics(true: np.ndarray, pred: np.ndarray) -> dict:
    err = pred - true
    ok = np.abs(err) <= np.maximum(2, 0.25 * true)
    r = float(np.corrcoef(true, pred)[0, 1]) if pred.std() > 0 else float("nan")
    return {"MAE": float(np.abs(err).mean()), "RMSE": float(np.sqrt((err ** 2).mean())),
            "bias": float(err.mean()), "within": float(ok.mean()), "r": r}


def bucket_mae(true: np.ndarray, pred: np.ndarray) -> dict:
    out = {}
    for name, lo, hi in (("0-5", 0, 5), ("6-20", 6, 20), ("21+", 21, 10 ** 6)):
        m = (true >= lo) & (true <= hi)
        out[name] = (int(m.sum()), float(np.abs(pred[m] - true[m]).mean()) if m.any() else float("nan"))
    return out


def classify(x: np.ndarray, thr) -> np.ndarray:
    return np.digitize(x, thr)  # x < thr[0] -> 0 (Low) ... same boundaries as congestion_from_density


def label_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    present = sorted(set(y))
    recalls = [float((p[y == c] == c).mean()) for c in present]
    f1s = []
    for c in present:
        tp = float(((p == c) & (y == c)).sum())
        prec = tp / max(1, (p == c).sum())
        rec = tp / max(1, (y == c).sum())
        f1s.append(0.0 if prec + rec == 0 else 2 * prec * rec / (prec + rec))
    return {"accuracy": float((p == y).mean()), "balanced_acc": float(np.mean(recalls)),
            "macro_f1": float(np.mean(f1s)), "off_by_2plus": float((np.abs(p - y) >= 2).mean())}


def confusion(y: np.ndarray, p: np.ndarray) -> np.ndarray:
    m = np.zeros((4, 4), dtype=int)
    for a, b in zip(y, p):
        m[a, b] += 1
    return m


# ---------------------------------------------------------------- threshold fitting
def fit_thresholds(x: np.ndarray, y: np.ndarray, n_candidates: int = 36):
    """Pick three ascending cut-offs maximising balanced accuracy (ties broken by accuracy)."""
    cands = np.unique(np.quantile(x, np.linspace(0.04, 0.99, n_candidates)))
    best, best_score = CURRENT_THRESHOLDS, -1.0
    for thr in itertools.combinations(cands, 3):
        p = classify(x, thr)
        m = label_metrics(y, p)
        score = m["balanced_acc"] + 0.01 * m["accuracy"]
        if score > best_score:
            best, best_score = thr, score
    return tuple(float(t) for t in best)


def cross_validate(x: np.ndarray, y: np.ndarray, folds: int = 5, repeats: int = 10, seed: int = 0):
    """Out-of-sample metrics for fitted thresholds: fit on k-1 folds, predict the held-out fold."""
    rng = np.random.default_rng(seed)
    rows, pooled = [], np.zeros((4, 4), dtype=int)
    for _ in range(repeats):
        idx = rng.permutation(len(x))
        pred = np.zeros(len(x), dtype=int)
        for f in range(folds):
            test = idx[f::folds]
            train = np.setdiff1d(idx, test)
            pred[test] = classify(x[test], fit_thresholds(x[train], y[train]))
        rows.append(label_metrics(y, pred))
        pooled += confusion(y, pred)
    df = pd.DataFrame(rows)
    return df.mean().to_dict(), df.std().to_dict(), pooled


# ---------------------------------------------------------------- reporting
def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = [floatfmt.format(v) if isinstance(v, float) else str(v) for v in row]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def md_confusion(m: np.ndarray) -> str:
    df = pd.DataFrame(m, columns=[f"pred {l}" for l in LEVELS])
    df.insert(0, "true", LEVELS)
    return md_table(df)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640, 1280])
    ap.add_argument("--tiled-models", nargs="*", default=["yolov8s.pt"],
                    help="models to also run with sliced (tiled) inference")
    ap.add_argument("--confs", type=float, nargs="+", default=[0.05, 0.08, 0.10, 0.15, 0.25, 0.35])
    args = ap.parse_args()

    labels = pd.read_csv(os.path.join(HERE, "labels.csv"))
    labels = labels[labels.usable == "yes"].reset_index(drop=True)
    ids = labels["id"].tolist()
    true_count = labels["count"].to_numpy(dtype=float)
    y = labels["congestion"].map(LEVELS.index).to_numpy()
    print(f"{len(ids)} usable labelled frames; class counts: "
          f"{ {l: int((y == i).sum()) for i, l in enumerate(LEVELS)} }")

    # 1. counting sweep
    rows, store = [], {}
    configs = [(w, sz, False) for w, sz in itertools.product(args.models, args.imgsz)]
    configs += [(w, 640, True) for w in args.tiled_models]
    for weights, imgsz, tiled in configs:
        dets = detect_all(weights, imgsz, ids, tiled=tiled)
        name = weights.replace(".pt", "") + ("+tiled" if tiled else "")
        for conf in args.confs:
            c, d = counts_and_density(dets, ids, conf)
            store[(name + ".pt", imgsz, conf)] = (c, d)
            rows.append({"model": name, "imgsz": imgsz, "conf": conf, **count_metrics(true_count, c)})
    sweep = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)

    baseline_key = ("yolov8s.pt", 640, 0.25)
    shipped_key = ("yolov8s.pt", 640, 0.08)
    settings = [("Before tuning", baseline_key), ("Shipped setting (full frame)", shipped_key)]
    tiled_rows = sweep[sweep.model.str.contains("tiled")]
    if len(tiled_rows):
        t = tiled_rows.iloc[0]
        settings.append(("Best sliced (tiled) setting", (t.model + ".pt", int(t.imgsz), float(t.conf))))
    settings.append(("Best setting by MAE", None))
    best_key = (sweep.loc[0, "model"] + ".pt", int(sweep.loc[0, "imgsz"]), float(sweep.loc[0, "conf"]))
    out = ["# CV evaluation results", "",
           f"Usable labelled frames: {len(ids)} (class counts: "
           + ", ".join(f"{l} {int((y == i).sum())}" for i, l in enumerate(LEVELS)) + ")", "",
           "## 1. Vehicle counting vs hand counts (sorted by MAE)", "",
           "`within` = share of frames where the count is within max(2, 25%) of the hand count.", "",
           md_table(sweep), ""]

    settings = [(t, k if k is not None else best_key) for t, k in settings]
    for title, key in settings:
        if key not in store:
            continue
        c, _ = store[key]
        b = bucket_mae(true_count, c)
        out += [f"### {title}: {key[0]} @ {key[1]}px, conf {key[2]}", "",
                "Count MAE by true count: " + ", ".join(f"{k}: {v[1]:.2f} (n={v[0]})" for k, v in b.items()), ""]

    # 2 + 3. congestion label from a per-frame feature (density, or vehicle count)
    majority = float((y == np.bincount(y).argmax()).mean())
    out += ["## 2. Congestion label vs hand-labelled class", "",
            f"Always answering the most common class ({LEVELS[int(np.bincount(y).argmax())]}) "
            f"scores accuracy {majority:.2f}; compare balanced accuracy (mean recall per class) "
            "and macro-F1, which that shortcut cannot game.", ""]
    summary = {}
    for title, key in settings:
        if key not in store:
            continue
        c, d = store[key]
        rows = []
        details = []
        for feat_name, x, thr in (("density (fixed 0.05/0.15/0.30)", d, CURRENT_THRESHOLDS), ("density (fitted, CV)", d, None),
                                  ("count (fitted, CV)", c.astype(float), None)):
            if thr is not None:
                m = label_metrics(y, classify(x, thr))
                off = m["off_by_2plus"]
                rows.append({"feature": feat_name, "accuracy": m["accuracy"], "balanced acc": m["balanced_acc"],
                             "macro-F1": m["macro_f1"], "off by 2+": off})
                details.append(("fixed", confusion(y, classify(x, thr)), thr))
            else:
                mean, std, pooled = cross_validate(x, y)
                fitted = fit_thresholds(x, y)
                rows.append({"feature": feat_name, "accuracy": mean["accuracy"], "balanced acc": mean["balanced_acc"],
                             "macro-F1": mean["macro_f1"], "off by 2+": mean["off_by_2plus"]})
                details.append((feat_name, pooled // 10, fitted))
        out += [f"### {title}: {key[0]} @ {key[1]}px, conf {key[2]}", "", md_table(pd.DataFrame(rows)), ""]
        for name, mat, thr in details[1:]:
            out += [f"Thresholds fitted on all frames for {name}: " + str(tuple(round(t, 3) for t in thr))
                    + " (shipped numbers above are the cross-validated ones)", ""]
        out += ["Confusion matrix, fixed density thresholds (rows = hand label):", "",
                md_confusion(details[0][1]), ""]
        summary[title] = {"key": list(map(str, key)), "rows": rows}

    os.makedirs(RESULTS, exist_ok=True)
    open(os.path.join(RESULTS, "report.md"), "w", encoding="utf-8").write("\n".join(out))
    json.dump({"sweep": sweep.to_dict("records"), "labels": summary, "majority_accuracy": majority},
              open(os.path.join(RESULTS, "summary.json"), "w"), indent=2)
    print("\n".join(out))


if __name__ == "__main__":
    main()
