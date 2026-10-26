"""Evaluate the violation rules on recorded clips (research/record_clip.py).

Three modes:

  replay     run every clip through the full pipeline (detector, tracker, calibration, violation rules),
             save what was flagged for human review, and report events per hour of video by type.
  score      read the review file (research/results/violation_review.csv) in which a person marked each
             saved event TP or FP, and report precision per violation type with a 95% Wilson interval.
  reversal   recall test for wrong-way driving: play a clip forward (so the flow is learned), then splice in
             a time-reversed stretch of the same footage, so real vehicles suddenly move against the
             learned flow. Reports how many reversed vehicles were flagged. A synthetic test; it says
             nothing about natural wrong-way events, which are far too rare to collect.

    python research/violation_eval.py replay --clips research/data/clips/*.mp4
    python research/violation_eval.py score
    python research/violation_eval.py reversal --clip research/data/clips/olive.mp4

Run from the project root. Events are never written to the app's challan log here.
"""
import argparse
import csv
import glob
import math
import os
import sys
import tempfile
import time

import cv2
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

RESULTS = os.path.join(HERE, "results")
REVIEW_DIR = os.path.join(RESULTS, "violation_events")
REVIEW_CSV = os.path.join(RESULTS, "violation_review.csv")


def run_clip(path, reference_mph=60.0, max_wait_s=3600):
    from live_monitor import LiveMonitor
    mon = LiveMonitor(path, os.path.basename(path), run_seconds=max_wait_s, reference_mph=reference_mph,
                      max_events=10 ** 6, min_event_gap_s=0.0, log_challans=False)
    mon.start()
    t0 = time.time()
    while mon.running and time.time() - t0 < max_wait_s:
        time.sleep(0.5)
        mon.snapshot()
    snap = mon.snapshot()
    return snap


def clip_seconds(path):
    cap = cv2.VideoCapture(path)
    n, fps = cap.get(cv2.CAP_PROP_FRAME_COUNT), cap.get(cv2.CAP_PROP_FPS) or 15.0
    cap.release()
    return n / fps


def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    w = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - w) / d, (c + w) / d


def cmd_replay(args):
    paths = sorted({p for pat in args.clips for p in glob.glob(pat)})
    if not paths:
        sys.exit("no clips found")
    os.makedirs(REVIEW_DIR, exist_ok=True)
    rows, review = [], []
    total_s = 0.0
    for path in paths:
        dur = clip_seconds(path)
        total_s += dur
        snap = run_clip(path)
        for i, ev in enumerate(snap["events"]):
            kind = ev.get("violation") or "speeding"
            name = f"{os.path.splitext(os.path.basename(path))[0]}_{i:03d}_{kind}.jpg"
            cv2.imwrite(os.path.join(REVIEW_DIR, name), ev["crop"])
            rows.append({"clip": os.path.basename(path), "type": kind, "time": ev["time"], "detail": ev.get("detail", ""),
                         "confidence": ev.get("confidence", ""), "image": name})
            review.append({"image": name, "type": kind, "judgement": ""})
        print(f"{os.path.basename(path)}: {dur:.0f} s, {len(snap['events'])} events, status {snap['status']} "
              f"({snap['message'] or 'ok'}), speed status {snap['speed_status']}", flush=True)
    ev_df = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    ev_df.to_csv(os.path.join(RESULTS, "violation_events.csv"), index=False)
    if review:
        new = pd.DataFrame(review)
        if os.path.exists(REVIEW_CSV):          # keep earlier judgements; only add images not reviewed yet
            old = pd.read_csv(REVIEW_CSV)
            new = pd.concat([old, new[~new["image"].isin(old["image"])]], ignore_index=True)
        new.to_csv(REVIEW_CSV, index=False)
        print(f"wrote {REVIEW_CSV}: fill the 'judgement' column with TP or FP after looking at each image in {REVIEW_DIR}")
    hours = total_s / 3600
    summary = (ev_df.groupby("type").size().rename("events").to_frame() if len(ev_df) else pd.DataFrame(columns=["events"]))
    summary["per_hour_of_video"] = summary["events"] / hours if hours else np.nan
    summary.to_csv(os.path.join(RESULTS, "violation_rates.csv"))
    print(f"\n{hours:.2f} h of video\n{summary.to_string()}")


def cmd_score(_args):
    if not os.path.exists(REVIEW_CSV):
        sys.exit("run replay first")
    df = pd.read_csv(REVIEW_CSV)
    df = df[df["judgement"].isin(["TP", "FP"])]
    rows = []
    for kind, g in df.groupby("type"):
        k, n = int((g["judgement"] == "TP").sum()), len(g)
        lo, hi = wilson(k, n)
        rows.append({"type": kind, "reviewed": n, "true_positives": k, "precision": k / n, "ci95_low": lo, "ci95_high": hi})
    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(RESULTS, "violation_precision.csv"), index=False)
    print(out.to_string(index=False, float_format=lambda v: f"{v:.2f}"))


def cmd_reversal(args):
    cap = cv2.VideoCapture(args.clip)
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    frames = []
    while True:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(f)
    cap.release()
    n_fwd = int(args.forward_s * fps)
    n_rev = int(args.reverse_s * fps)
    if len(frames) < n_fwd + n_rev:
        sys.exit(f"clip too short: need {args.forward_s + args.reverse_s:.0f} s")
    seg = frames[n_fwd - n_rev:n_fwd][::-1]          # the last `reverse_s` seconds before the splice, played backwards
    spliced = frames[:n_fwd] + seg
    h, w = spliced[0].shape[:2]
    tmp = os.path.join(tempfile.gettempdir(), "reversal_test.mp4")
    wr = cv2.VideoWriter(tmp, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in spliced:
        wr.write(f)
    wr.release()
    snap = run_clip(tmp)
    wrong = [e for e in snap["events"] if e.get("violation") == "wrong_way"]
    other = [e for e in snap["events"] if e.get("violation") not in (None, "wrong_way")]
    print(f"{len(wrong)} wrong-way events in the reversed {args.reverse_s:.0f} s "
          f"(forward part {args.forward_s:.0f} s teaches the flow); {len(other)} other violation events; speed status {snap['speed_status']}")
    os.makedirs(RESULTS, exist_ok=True)
    pd.DataFrame([{"clip": os.path.basename(args.clip), "forward_s": args.forward_s, "reverse_s": args.reverse_s,
                   "wrong_way_events": len(wrong), "other_events": len(other)}]).to_csv(
        os.path.join(RESULTS, "violation_reversal.csv"), mode="a", index=False,
        header=not os.path.exists(os.path.join(RESULTS, "violation_reversal.csv")))


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay")
    r.add_argument("--clips", nargs="+", required=True)
    sub.add_parser("score")
    v = sub.add_parser("reversal")
    v.add_argument("--clip", required=True)
    v.add_argument("--forward-s", type=float, default=45.0)
    v.add_argument("--reverse-s", type=float, default=15.0)
    args = ap.parse_args()
    {"replay": cmd_replay, "score": cmd_score, "reversal": cmd_reversal}[args.cmd](args)


if __name__ == "__main__":
    main()
