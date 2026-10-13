"""Can a small learned classifier beat the fixed density thresholds at the congestion label?

Features come from the cached YOLO detections of each labelled frame (count, density, box-size
statistics), the labels are the hand-labelled congestion classes. Models are scored with repeated
5-fold cross-validation (fit on 4/5 of the frames, predict the held-out 1/5), the same protocol as
`evaluate_cv.py`, and compared with the app's fixed thresholds (0.05 / 0.15 / 0.30 on density) on
the same frames. With 124 frames and only 10 High/Severe ones this is a small experiment: the point
is an honest answer, not a headline number.

    python cv_eval/learned_classifier.py
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import evaluate_cv as E  # noqa: E402


def features(dets: list, conf: float) -> list[float]:
    areas = np.array([a for c, a in dets if c >= conf]) if dets else np.array([])
    if len(areas) == 0:
        return [0, 0, 0, 0, 0, 0]
    small = float((areas < 0.0005).sum())
    return [len(areas), float(min(1.0, areas.sum())), float(np.median(areas)), float(areas.max()),
            small, float(len(areas) - small)]


def repeated_cv(make_model, X, y, folds=5, repeats=10, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(repeats):
        idx = rng.permutation(len(X))
        pred = np.zeros(len(X), dtype=int)
        for f in range(folds):
            test = idx[f::folds]
            train = np.setdiff1d(idx, test)
            if len(set(y[train])) < 2:
                pred[test] = y[train][0]
                continue
            model = make_model()
            model.fit(X[train], y[train])
            pred[test] = model.predict(X[test])
        rows.append(E.label_metrics(y, pred))
    df = pd.DataFrame(rows)
    return df.mean().to_dict(), df.std().to_dict()


def main():
    labels = pd.read_csv(os.path.join(HERE, "labels.csv"))
    labels = labels[labels.usable == "yes"].reset_index(drop=True)
    ids = labels["id"].tolist()
    y = labels["congestion"].map(E.LEVELS.index).to_numpy()
    out = ["# Learned congestion classifier vs fixed thresholds", "",
           f"{len(ids)} usable frames; classes " + ", ".join(f"{l} {int((y == i).sum())}" for i, l in enumerate(E.LEVELS)),
           "Repeated 5-fold cross-validation x10 (out-of-sample).", ""]
    for name, tiled, conf in (("full frame, conf 0.08", False, 0.08), ("sliced (tiled), conf 0.08", True, 0.08)):
        cache = os.path.join(E.DET_DIR, f"yolov8s_640{'_tiled' if tiled else ''}_c{E.DETECT_CONF}.json")
        if not os.path.exists(cache):
            continue
        dets = json.load(open(cache))
        X = np.array([features(dets[i], conf) for i in ids])
        fixed = E.label_metrics(y, E.classify(X[:, 1], E.CURRENT_THRESHOLDS))
        rows = [{"model": "fixed density thresholds (shipped)", **{k: fixed[k] for k in ("accuracy", "balanced_acc", "macro_f1")},
                 "std (bal. acc)": 0.0}]
        makers = {
            "logistic regression": lambda: make_pipeline(StandardScaler(), LogisticRegression(
                max_iter=2000, class_weight="balanced")),
            "random forest": lambda: RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                           class_weight="balanced", random_state=0),
            "gradient boosting": lambda: GradientBoostingClassifier(n_estimators=80, max_depth=2, random_state=0),
        }
        for mname, make in makers.items():
            mean, std = repeated_cv(make, X, y)
            rows.append({"model": mname, "accuracy": mean["accuracy"], "balanced_acc": mean["balanced_acc"],
                         "macro_f1": mean["macro_f1"], "std (bal. acc)": std["balanced_acc"]})
        out += [f"## Detections: {name}", "", E.md_table(pd.DataFrame(rows)), ""]
    text = "\n".join(out)
    open(os.path.join(E.RESULTS, "learned_classifier.md"), "w", encoding="utf-8").write(text)
    print(text)


if __name__ == "__main__":
    main()
