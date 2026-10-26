"""Log time-stamped vision measurements from live Caltrans cameras, aligned to 5-minute bins.

This builds the camera side of the ground-truth validation (research/pems_validation.py): PeMS loop
detectors report flow, occupancy and speed in 5-minute bins, so every round here starts on a
5-minute wall-clock boundary.

Each round, for every selected camera:
  * fetch the current snapshot, run YOLOv8s, store vehicle count and image density
    (an unavailable-placeholder frame is logged as unavailable, never as an empty road);
  * optionally (--burst-every N rounds) watch ~55 s of the live video through the same tracker used by
    the app and store the median RAW ground-plane speed (before any scaling to a typical speed) and the
    camera-geometry fit, for the speed validation.
  * optionally (--save-frames) keep the JPEG, which is the pool the detector evaluation labels from.

    python research/log_cameras.py --district 7 --route-filter I-5 I-405 --max-cameras 8 --hours 24 \
        --burst-every 3 --save-frames

Output: research/data/camera_log.csv (appended), research/data/camera_set.json (the chosen cameras),
research/data/frames/<camera>_<utc>.jpg. Leave it running; it is safe to stop and restart.
"""
import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from cv_module import CameraUnavailableError, analyze_frame, congestion_from_density, load_frame  # noqa: E402
from live_cameras import fetch_cameras  # noqa: E402
from live_video import is_playable_url  # noqa: E402

DATA = os.path.join(HERE, "data")
LOG_PATH = os.path.join(DATA, "camera_log.csv")
SET_PATH = os.path.join(DATA, "camera_set.json")
PT = ZoneInfo("America/Los_Angeles")
FIELDS = ["utc", "pacific", "camera_id", "name", "route", "latitude", "longitude", "status", "vehicle_count",
          "density", "level", "burst_tracks", "raw_speed_median_mph", "scale", "camera_height_m", "fit_corr"]


def camera_id(cam: dict) -> str:
    return cam["image_url"].rstrip("/").split("/")[-1].rsplit(".", 1)[0]


def choose_cameras(district, route_filter, max_cameras, need_video):
    cams = fetch_cameras(district)
    cams = [c for c in cams if c.get("latitude") and c.get("longitude")]
    if route_filter:
        cams = [c for c in cams if any(r.lower() in (c["name"] + " " + c["route"]).lower() for r in route_filter)]
    if need_video:
        cams = [c for c in cams if is_playable_url(c.get("stream_url", ""))]
    out = []
    for c in cams:                       # keep only cameras that currently give a real picture
        try:
            frame = load_frame(c["image_url"])
            analyze_frame(frame)
        except CameraUnavailableError:
            continue
        except Exception:
            continue
        out.append(c)
        if len(out) >= max_cameras:
            break
    return out


def burst(cam, seconds=55):
    """Watch the live video briefly; return the raw (unscaled) median speed and the camera fit."""
    from live_monitor import LiveMonitor
    mon = LiveMonitor(cam["stream_url"], cam["name"], run_seconds=seconds, detect_violations=False)
    mon.start()
    best = {"tracks": 0, "raw": None, "scale": None, "calib": None}
    while mon.running:
        time.sleep(1.0)
        snap = mon.snapshot()
        if snap["raw_speed_median"] is not None:
            best.update(tracks=snap["tracks_with_speed"], raw=snap["raw_speed_median"], scale=snap["scale"])
        if snap["calibration"]:
            best["calib"] = snap["calibration"]
    return best


def write_row(row):
    new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--district", type=int, default=7)
    ap.add_argument("--route-filter", nargs="*", default=["I-5", "I-405", "I-10", "I-110", "US-101"])
    ap.add_argument("--max-cameras", type=int, default=8)
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--interval-min", type=int, default=5)
    ap.add_argument("--burst-every", type=int, default=0, help="video burst every N rounds (0 = never)")
    ap.add_argument("--save-frames", action="store_true")
    ap.add_argument("--rebuild-set", action="store_true")
    args = ap.parse_args()
    os.makedirs(DATA, exist_ok=True)

    if os.path.exists(SET_PATH) and not args.rebuild_set:
        cams = json.load(open(SET_PATH, encoding="utf-8"))
    else:
        cams = choose_cameras(args.district, args.route_filter, args.max_cameras, need_video=args.burst_every > 0)
        json.dump(cams, open(SET_PATH, "w", encoding="utf-8"), indent=2)
    print(f"{len(cams)} cameras: " + "; ".join(c["name"] for c in cams), flush=True)
    if args.save_frames:
        os.makedirs(os.path.join(DATA, "frames"), exist_ok=True)

    end = time.time() + args.hours * 3600
    rnd = 0
    while time.time() < end:
        step = args.interval_min * 60
        time.sleep(max(0, step - (time.time() % step)))      # wait for the next 5-minute boundary
        rnd += 1
        now = datetime.now(timezone.utc)
        for cam in cams:
            cid = camera_id(cam)
            row = {"utc": now.strftime("%Y-%m-%dT%H:%M:%SZ"), "pacific": now.astimezone(PT).strftime("%Y-%m-%d %H:%M"),
                   "camera_id": cid, "name": cam["name"], "route": cam["route"], "latitude": cam["latitude"],
                   "longitude": cam["longitude"], "status": "ok"}
            try:
                frame = load_frame(cam["image_url"])
                res = analyze_frame(frame)
                row.update(vehicle_count=res["vehicle_count"], density=res["density_ratio"],
                           level=congestion_from_density(res["density_ratio"]))
                if args.save_frames:
                    cv2.imwrite(os.path.join(DATA, "frames", f"{cid}_{now.strftime('%Y%m%dT%H%M%SZ')}.jpg"), frame,
                                [cv2.IMWRITE_JPEG_QUALITY, 90])
            except CameraUnavailableError:
                row["status"] = "unavailable"
            except Exception as e:
                row["status"] = f"error: {type(e).__name__}"
            if args.burst_every and rnd % args.burst_every == 0 and row["status"] == "ok" and is_playable_url(cam.get("stream_url", "")):
                try:
                    b = burst(cam)
                    row.update(burst_tracks=b["tracks"], raw_speed_median_mph=b["raw"], scale=b["scale"],
                               camera_height_m=(b["calib"] or {}).get("camera_height_m"), fit_corr=(b["calib"] or {}).get("correlation"))
                except Exception as e:
                    print(f"burst failed for {cam['name']}: {e}", flush=True)
            write_row(row)
        print(f"{datetime.now():%H:%M:%S} round {rnd} logged ({len(cams)} cameras)", flush=True)


if __name__ == "__main__":
    main()
