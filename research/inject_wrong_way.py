"""Recall test for the wrong-way rule: drive a real car backwards along a real lane, over real footage.

Natural wrong-way events are far too rare to collect, and reversing a whole clip is unrealistic (every vehicle then
goes the "wrong" way, which the rule correctly treats as a two-way road). Instead:

  1. detect and track vehicles in the first part of a recorded clip and pick long, clear vehicle paths;
  2. cut that vehicle's image out of the frame where it is largest;
  3. later in the clip, paste the image frame by frame along the same path played backwards, sized to the original boxes,
     so one vehicle travels against the traffic of its own lane while everything else stays real;
  4. run the full pipeline (detector, tracker, rules) on the result and count which injected cars were flagged.

Reports recall (flagged injections / injections) and wrong-way flags outside the injection windows (false alarms).
The pasted car is a real vehicle image but is not a perfect composite (no shadows or occlusion), so treat the number as
an upper bound on detection difficulty for clear cases.

    python research/inject_wrong_way.py research/data/clips/i5_olive_*.mp4 --injections 3
"""
import argparse
import glob
import os
import sys
import tempfile

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))
sys.path.insert(0, HERE)

from vehicle_tracking import CentroidTracker  # noqa: E402

RESULTS = os.path.join(HERE, "results")


def clip_info(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, f = cap.read()
    cap.release()
    return fps, n, f.shape[:2]


def find_paths(path, fps, t_from, t_to, n_paths, img_h):
    """Track vehicles in [t_from, t_to] (streaming, sampled at ~3.7 Hz); return up to n_paths long clear paths, each
    with the image of the vehicle cut from the frame where it was largest."""
    from ultralytics import YOLO
    from cv_module import VEHICLE_CLASS_IDS
    model = YOLO("yolov8s.pt")
    tracker = CentroidTracker(max_history_s=60.0)
    step = max(1, int(round(0.27 * fps)))
    boxes, sprite = {}, {}
    cap = cv2.VideoCapture(path)
    i = -1
    while True:
        ok = cap.grab()
        i += 1
        if not ok or i > t_to * fps:
            break
        if i < t_from * fps or i % step:
            continue
        ok, frame = cap.retrieve()
        if not ok:
            continue
        res = model.predict(frame, conf=0.35, imgsz=640, classes=list(VEHICLE_CLASS_IDS), verbose=False)[0]
        dets = [{"cls_name": "car", "conf": float(b.conf.item()), "box": tuple(b.xyxy[0].tolist())} for b in res.boxes]
        for tr in tracker.update(i / fps, dets):
            boxes.setdefault(tr.track_id, []).append((i / fps, tr.last_box))
            x1, y1, x2, y2 = (int(v) for v in tr.last_box)
            fh, fw = frame.shape[:2]
            whole = x1 > 2 and y1 > 2 and x2 < fw - 2 and y2 < fh - 4 and (y2 - y1) >= 0.45 * (x2 - x1)   # not cut by the edge
            if whole and (tr.track_id not in sprite or (x2 - x1) > sprite[tr.track_id].shape[1]):
                crop = frame[max(0, y1):y2, max(0, x1):x2]
                if crop.size:
                    sprite[tr.track_id] = crop.copy()
    cap.release()
    cands = []
    for tid, pth in boxes.items():
        if len(pth) < 10 or tid not in sprite:
            continue
        ys = [b[1][3] for b in pth]
        widths = [b[1][2] - b[1][0] for b in pth]
        if max(ys) - min(ys) >= 0.25 * img_h and max(widths) >= 28:
            cands.append((max(ys) - min(ys), pth, sprite[tid]))
    cands.sort(key=lambda c: -c[0])
    return [(c[1], c[2]) for c in cands[:n_paths]]


def box_at(path, t):
    ts = np.array([p[0] for p in path])
    bx = np.array([p[1] for p in path], dtype=float)
    return [float(np.interp(t, ts, bx[:, k])) for k in range(4)]


def paste(frame, sprite, box):
    h, w = frame.shape[:2]
    bx1, by1, bx2, by2 = box
    bw, bh = int(bx2 - bx1), int(by2 - by1)
    if bw < 6 or bh < 6:
        return
    spr = cv2.resize(sprite, (bw, bh))
    ax, ay = int(bx1), int(by1)
    ax2, ay2 = min(w, ax + bw), min(h, ay + bh)
    if ax < 0 or ay < 0 or ax2 <= ax or ay2 <= ay:
        return
    frame[ay:ay2, ax:ax2] = spr[: ay2 - ay, : ax2 - ax]


def write_injected(path, out, fps, size, plan):
    """Stream the clip to `out`, pasting each planned vehicle backwards along its path. plan: [(start_t, path, sprite)]."""
    h, w = size
    cap = cv2.VideoCapture(path)
    wr = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = i / fps
        for start, pth, spr in plan:
            t0, t1 = pth[0][0], pth[-1][0]
            if start <= t <= start + (t1 - t0):
                paste(frame, spr, box_at(pth, t1 - (t - start)))      # backwards along the path
        wr.write(frame)
        i += 1
    cap.release()
    wr.release()


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("clips", nargs="+")
    ap.add_argument("--injections", type=int, default=3)
    ap.add_argument("--learn-s", type=float, default=60.0, help="untouched footage before the first injection")
    ap.add_argument("--spacing-s", type=float, default=55.0)
    args = ap.parse_args()
    from violation_eval import run_clip
    rows = []
    for clip in sorted({p for pat in args.clips for p in glob.glob(pat)}):
        fps, n_frames, size = clip_info(clip)
        paths = find_paths(clip, fps, 5.0, args.learn_s, args.injections, size[0])
        if not paths:
            print(f"{os.path.basename(clip)}: no clear vehicle path found; skipped")
            continue
        plan, windows = [], []
        for k in range(args.injections):
            start = args.learn_s + k * args.spacing_s
            pth, spr = paths[k % len(paths)]
            if (start + (pth[-1][0] - pth[0][0])) * fps >= n_frames:
                break
            plan.append((start, pth, spr))
            windows.append((start, start + pth[-1][0] - pth[0][0]))
        tmp = os.path.join(tempfile.gettempdir(), "inject_test.mp4")
        write_injected(clip, tmp, fps, size, plan)
        snap = run_clip(tmp)
        ww = [e for e in snap["events"] if e.get("violation") == "wrong_way"]
        hits = sum(1 for a, b in windows if any(a <= e["stream_t"] <= b + 3.0 for e in ww))
        outside = sum(1 for e in ww if not any(a <= e["stream_t"] <= b + 3.0 for a, b in windows))
        rows.append({"clip": os.path.basename(clip), "injections": len(windows), "flagged": hits, "false_alarms": outside})
        print(f"{os.path.basename(clip)}: {hits}/{len(windows)} injected wrong-way cars flagged, {outside} wrong-way flags elsewhere")
    if rows:
        df = pd.DataFrame(rows)
        os.makedirs(RESULTS, exist_ok=True)
        df.to_csv(os.path.join(RESULTS, "wrong_way_injection.csv"), index=False)
        print(f"total: {df['flagged'].sum()}/{df['injections'].sum()} flagged, {df['false_alarms'].sum()} false alarms")


if __name__ == "__main__":
    main()
