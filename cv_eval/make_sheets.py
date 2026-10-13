"""Build contact sheets of the collected frames for hand-labelling and review.

Each frame is drawn with its id burned into the corner. With --labels, the
current count and congestion class from cv_eval/labels.csv are overlaid too, so
a reviewer can check the labels against the images quickly.

    python cv_eval/make_sheets.py                 # plain sheets for labelling
    python cv_eval/make_sheets.py --labels        # sheets with labels overlaid
"""
import argparse
import csv
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))


def read_labels(path):
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", action="store_true", help="overlay labels from labels.csv")
    parser.add_argument("--per-sheet", type=int, default=6)
    parser.add_argument("--width", type=int, default=640, help="width of each frame in the sheet")
    parser.add_argument("--start", default=None, help="first frame id to include, e.g. f001")
    args = parser.parse_args()

    with open(os.path.join(HERE, "frames.csv"), newline="", encoding="utf-8") as f:
        ids = [r["id"] for r in csv.DictReader(f)]
    if args.start:
        ids = [i for i in ids if i >= args.start]
    labels = read_labels(os.path.join(HERE, "labels.csv")) if args.labels else {}

    out_dir = os.path.join(HERE, "sheets_labelled" if args.labels else "sheets")
    os.makedirs(out_dir, exist_ok=True)
    cols = 2
    for s in range(0, len(ids), args.per_sheet):
        tiles = []
        for fid in ids[s: s + args.per_sheet]:
            img = cv2.imread(os.path.join(HERE, "frames", f"{fid}.jpg"))
            h, w = img.shape[:2]
            img = cv2.resize(img, (args.width, int(h * args.width / w)))
            text = fid
            if fid in labels:
                text += f"  n={labels[fid]['count']}  {labels[fid]['congestion']}  use={labels[fid]['usable']}"
            cv2.rectangle(img, (0, 0), (img.shape[1], 26), (0, 0, 0), -1)
            cv2.putText(img, text, (6, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            tiles.append(img)
        th = max(t.shape[0] for t in tiles)
        padded = [cv2.copyMakeBorder(t, 0, th - t.shape[0], 0, 0, cv2.BORDER_CONSTANT) for t in tiles]
        while len(padded) % cols:
            padded.append(np.zeros_like(padded[0]))
        rows = [np.hstack(padded[i: i + cols]) for i in range(0, len(padded), cols)]
        sheet = np.vstack(rows)
        cv2.imwrite(os.path.join(out_dir, f"sheet_{s // args.per_sheet + 1:02d}.jpg"), sheet,
                    [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Wrote sheets to {out_dir}")


if __name__ == "__main__":
    main()
