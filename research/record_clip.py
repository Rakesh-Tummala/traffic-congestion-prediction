"""Record a short clip from a live Caltrans HLS stream to disk, so the violation rules can be evaluated on
exactly the same footage every time (live streams cannot be replayed).

    python research/record_clip.py --district 7 --name "(37) Olive" --seconds 180
    python research/record_clip.py --url https://.../playlist.m3u8 --seconds 300 --label olive_am_peak

Clips go to research/data/clips/<label>_<utc>.mp4. Choose camera and time of day with care: wrong-way and
stopped-vehicle events are rare, so several hours of footage across different cameras are needed for a
meaningful false-positive rate. Run from the project root so the model weights resolve.
"""
import argparse
import os
import sys
import time
from datetime import datetime, timezone

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from live_cameras import fetch_cameras  # noqa: E402

CLIPS = os.path.join(HERE, "data", "clips")


def record(url: str, seconds: float, out_path: str) -> dict:
    cap = cv2.VideoCapture()
    cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
    cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
    if not cap.open(url):
        raise RuntimeError("could not open the stream")
    fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("stream returned no frames")
    h, w = frame.shape[:2]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    n, t0 = 0, time.time()
    while n / fps < seconds and time.time() - t0 < 4 * seconds + 60:   # stop at `seconds` of VIDEO (HLS can run ahead of real time)
        writer.write(frame)
        n += 1
        ok, frame = cap.read()
        if not ok:
            break
    writer.release()
    cap.release()
    return {"frames": n, "fps": fps, "seconds_of_video": n / fps, "size": (w, h)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--district", type=int, default=7)
    ap.add_argument("--name", help="part of the camera name, e.g. '(37) Olive'")
    ap.add_argument("--url")
    ap.add_argument("--seconds", type=float, default=180)
    ap.add_argument("--label", default=None)
    args = ap.parse_args()
    url, label = args.url, args.label
    if not url:
        cams = [c for c in fetch_cameras(args.district) if args.name and args.name.lower() in c["name"].lower() and c["stream_url"]]
        if not cams:
            sys.exit("no camera with a video stream matches that name")
        url = cams[0]["stream_url"]
        label = label or cams[0]["name"].replace(" ", "").replace(":", "").replace("(", "").replace(")", "")
    label = label or "clip"
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out = os.path.join(CLIPS, f"{label}_{stamp}.mp4")
    info = record(url, args.seconds, out)
    print(f"saved {out}: {info['seconds_of_video']:.0f} s of video at {info['fps']:.0f} fps, {info['size']}")


if __name__ == "__main__":
    main()
