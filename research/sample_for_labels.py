"""Pick a time-stratified sample of the logged camera frames to hand-count, and build contact sheets.

research/log_cameras.py --save-frames keeps a JPEG per camera every 5 minutes. This script draws the same number
of usable frames from each time-of-day bucket (Pacific time), so the evaluation cannot be all daytime, and writes
  research/data/label_set.csv      id, file, camera, pacific, bucket   (the evaluation set)
  research/data/labels_new.csv     id, count, usable, note             (empty template to fill in)
  research/data/sheets/*.jpg       contact sheets, six frames each, id burned in

    python research/sample_for_labels.py --per-bucket 100

Counting rule (write it down in the paper): count every vehicle you can identify as a car, truck, bus or
motorcycle, including small distant ones; mark usable=no for placeholders, glare, a camera pointed away from the road.
For research-grade labels two people should count a subset independently (see research/label_agreement.py).
"""
import argparse
import csv
import glob
import os
import random
import sys
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from cv_module import is_unavailable_frame  # noqa: E402

DATA = os.path.join(HERE, "data")
PT = ZoneInfo("America/Los_Angeles")
BUCKETS = [("night", 21, 5), ("dawn", 5, 7), ("morning", 7, 10), ("midday", 10, 15), ("afternoon", 15, 19), ("dusk", 19, 21)]


def bucket_of(hour: int) -> str:
    for name, a, b in BUCKETS:
        if (a <= hour < b) if a < b else (hour >= a or hour < b):
            return name
    return "other"


def parse_name(path):
    base = os.path.splitext(os.path.basename(path))[0]
    cid, stamp = base.rsplit("_", 1)
    utc = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
    return cid, utc.astimezone(PT)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--per-bucket", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--per-sheet", type=int, default=6)
    args = ap.parse_args()
    files = glob.glob(os.path.join(DATA, "frames", "*.jpg"))
    by_bucket = {}
    for f in files:
        cid, pt = parse_name(f)
        by_bucket.setdefault(bucket_of(pt.hour), []).append((f, cid, pt))
    rng = random.Random(args.seed)
    chosen = []
    for name, _, _ in BUCKETS:
        pool = by_bucket.get(name, [])
        rng.shuffle(pool)
        got = 0
        for f, cid, pt in pool:
            if got >= args.per_bucket:
                break
            if is_unavailable_frame(cv2.imread(f)):
                continue
            chosen.append((f, cid, pt, name))
            got += 1
        print(f"{name:10s} available {len(pool):5d}  chosen {got}")
    chosen.sort(key=lambda r: (r[3], r[2]))
    os.makedirs(os.path.join(DATA, "sheets"), exist_ok=True)
    with open(os.path.join(DATA, "label_set.csv"), "w", newline="", encoding="utf-8") as f1, \
            open(os.path.join(DATA, "labels_new.csv"), "w", newline="", encoding="utf-8") as f2:
        w1, w2 = csv.writer(f1), csv.writer(f2)
        w1.writerow(["id", "file", "camera", "pacific", "bucket"])
        w2.writerow(["id", "count", "usable", "note"])
        for i, (f, cid, pt, name) in enumerate(chosen, 1):
            fid = f"n{i:03d}"
            w1.writerow([fid, os.path.relpath(f, DATA).replace("\\", "/"), cid, pt.strftime("%Y-%m-%d %H:%M"), name])
            w2.writerow([fid, "", "", ""])
    for s in range(0, len(chosen), args.per_sheet):
        tiles = []
        for k, (f, cid, pt, name) in enumerate(chosen[s: s + args.per_sheet], start=s + 1):
            img = cv2.imread(f)
            h, w = img.shape[:2]
            img = cv2.resize(img, (640, int(h * 640 / w)))
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, f"n{k:03d}  {name}", (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(img)
        th = max(t.shape[0] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(tiles) % 2:
            tiles.append(np.zeros_like(tiles[0]))
        sheet = np.vstack([np.hstack(tiles[i: i + 2]) for i in range(0, len(tiles), 2)])
        cv2.imwrite(os.path.join(DATA, "sheets", f"sheet_{s // args.per_sheet + 1:03d}.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
    print(f"{len(chosen)} frames in label_set.csv; template labels_new.csv; sheets in {os.path.join(DATA, 'sheets')}")


if __name__ == "__main__":
    main()
