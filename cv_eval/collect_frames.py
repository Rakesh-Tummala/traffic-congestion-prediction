"""Download a stratified sample of live Caltrans camera frames to label for CV evaluation.

Takes the same number of random in-service cameras from every district, so the
set covers different regions, camera heights and lighting rather than one city.
Frames are saved as cv_eval/frames/<id>.jpg and indexed in cv_eval/frames.csv
(id, district, camera name, place, source URL, fetch time).

    python cv_eval/collect_frames.py --per-district 10 --seed 0
"""
import argparse
import csv
import os
import random
import sys
from datetime import datetime

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from cv_module import load_frame  # noqa: E402
from live_cameras import DISTRICTS, fetch_cameras  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--per-district", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--districts", type=int, nargs="+", default=None,
                        help="only these districts (default: all)")
    parser.add_argument("--name-prefix", nargs="+", default=None,
                        help="only cameras whose name starts with one of these, e.g. I- US-")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    frames_dir = os.path.join(HERE, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    index_path = os.path.join(HERE, "frames.csv")

    existing = set()
    if os.path.exists(index_path):
        with open(index_path, newline="", encoding="utf-8") as f:
            existing = {row["image_url"] for row in csv.DictReader(f)}
    new_file = not os.path.exists(index_path)

    saved = 0
    with open(index_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if new_file:
            writer.writerow(["id", "district", "camera", "place", "image_url", "fetched_at"])
        next_id = len(existing) + 1
        for d in (args.districts or DISTRICTS):
            try:
                cams = [c for c in fetch_cameras(d) if c["image_url"] not in existing]
            except Exception as e:
                print(f"District {d}: skipped ({e})")
                continue
            if args.name_prefix:
                cams = [c for c in cams if c["name"].startswith(tuple(args.name_prefix))]
            rng.shuffle(cams)
            got = 0
            for cam in cams:
                if got >= args.per_district:
                    break
                try:
                    frame = load_frame(cam["image_url"])
                except Exception:
                    continue
                fid = f"f{next_id:03d}"
                cv2.imwrite(os.path.join(frames_dir, f"{fid}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                writer.writerow([fid, d, cam["name"], cam["nearby_place"], cam["image_url"],
                                 datetime.now().isoformat(timespec="seconds")])
                existing.add(cam["image_url"])
                next_id += 1
                got += 1
                saved += 1
            print(f"District {d}: saved {got}")
    print(f"Saved {saved} new frames to {frames_dir}")


if __name__ == "__main__":
    main()
