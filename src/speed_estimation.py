"""Estimate vehicle speeds from a short live-video burst and flag speeding
relative to a posted limit.

Requires the camera's HLS video stream, not the static JPEG snapshot used
elsewhere in this project — Caltrans's snapshot only refreshes ~once a
minute (confirmed against the live feed), far too coarse to see a vehicle
move between two "live" fetches. The stream URL comes from
`live_cameras.fetch_cameras`'s `stream_url` field.

Calibration caveat: converting pixel displacement to a real-world speed
needs a real-world scale (meters per pixel), and there is no camera
calibration data published for these cameras. This module takes that scale
as an explicit input (`meters_per_pixel`) rather than guessing — the caller
is expected to derive it from a visible reference of known size (e.g. a
lane's standard width), documented in the README. Treat resulting speeds as
an approximation, not a certified measurement.
"""
import argparse
import time

import cv2
import numpy as np

from cv_module import VEHICLE_CLASS_IDS, get_model

US_LANE_WIDTH_M = 3.7  # standard US freeway lane width, used as a default calibration reference
MPS_TO_MPH = 2.2369362920544
MPS_TO_KMH = 3.6


def capture_frame_burst(stream_url: str, num_frames: int = 6, sample_every: int = 4, timeout_s: float = 15.0):
    """Read frames from a live HLS stream, keeping every `sample_every`-th
    decoded frame. Timestamps come from the stream's own internal clock
    (CAP_PROP_POS_MSEC), not wall-clock read time: an HLS source can hand
    over an already-buffered segment far faster than real time, so measuring
    elapsed time via time.time() between reads massively overestimates
    speed (confirmed during development — reads of a ~1.5s burst completed
    in under 20ms of wall-clock time)."""
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise ValueError(f"Could not open video stream: {stream_url}")

    frames = []
    i = 0
    deadline = time.time() + timeout_s
    try:
        while len(frames) < num_frames and time.time() < deadline:
            ok, frame = cap.read()
            if not ok:
                break
            if i % sample_every == 0:
                pos_s = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                frames.append((frame.copy(), pos_s))
            i += 1
    finally:
        cap.release()

    if len(frames) < 2:
        raise ValueError(f"Only captured {len(frames)} frame(s) from the stream; need at least 2 to estimate speed.")
    return frames


def detect_vehicles(frame: np.ndarray, conf: float = 0.25) -> list[dict]:
    model = get_model()
    results = model.predict(frame, conf=conf, classes=list(VEHICLE_CLASS_IDS.keys()), verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls_id = int(box.cls.item())
        name = VEHICLE_CLASS_IDS.get(cls_id)
        if name is None:
            continue
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        detections.append({
            "cls_name": name, "conf": float(box.conf.item()),
            "box": (x1, y1, x2, y2), "centroid": ((x1 + x2) / 2, (y1 + y2) / 2),
        })
    return detections


def track_across_frames(frame_detections: list[list[dict]], max_match_dist_px: float = 80.0) -> list[list[dict]]:
    """Greedy nearest-centroid matching between consecutive frames' detections.

    This is deliberately simple — not a real multi-object tracker (no
    re-identification after a missed frame, no appearance model, no
    handling of two vehicles crossing paths). It's adequate for a short
    (4-6 frame) burst with a handful of well-separated vehicles, which is
    the scenario this feature targets; it will misassociate vehicles in
    dense, fast-crossing traffic.
    """
    tracks: list[list[dict]] = []
    active: list[int] = []

    for frame_idx, dets in enumerate(frame_detections):
        for d in dets:
            d["frame_idx"] = frame_idx
        unmatched = list(dets)
        new_active = []
        for track_i in active:
            last = tracks[track_i][-1]
            best_j, best_dist = None, max_match_dist_px
            for j, d in enumerate(unmatched):
                dist = float(np.hypot(d["centroid"][0] - last["centroid"][0], d["centroid"][1] - last["centroid"][1]))
                if dist < best_dist:
                    best_j, best_dist = j, dist
            if best_j is not None:
                tracks[track_i].append(unmatched.pop(best_j))
                new_active.append(track_i)
        for d in unmatched:
            tracks.append([d])
            new_active.append(len(tracks) - 1)
        active = new_active

    return [t for t in tracks if len(t) >= 2]


def estimate_speed_mps(track: list[dict], frame_times: list[float], meters_per_pixel: float) -> float | None:
    """Average speed (m/s) across consecutive matched points in a track."""
    if len(track) < 2:
        return None
    speeds = []
    for a, b in zip(track, track[1:]):
        dt = frame_times[b["frame_idx"]] - frame_times[a["frame_idx"]]
        if dt <= 0:
            continue
        dist_px = float(np.hypot(b["centroid"][0] - a["centroid"][0], b["centroid"][1] - a["centroid"][1]))
        speeds.append(dist_px * meters_per_pixel / dt)
    return float(np.mean(speeds)) if speeds else None


def flag_speeding(speed_mph: float, limit_mph: float, margin_mph: float = 5.0):
    """Flag as speeding only if over the limit by more than `margin_mph`,
    to absorb calibration/tracking noise rather than flagging on tiny
    overshoots the estimate can't actually resolve reliably."""
    over_mph = speed_mph - limit_mph
    return over_mph > margin_mph, over_mph


def check_speeds(stream_url: str, meters_per_pixel: float, limit_mph: float,
                  num_frames: int = 6, sample_every: int = 4, conf: float = 0.25) -> dict:
    frames_with_t = capture_frame_burst(stream_url, num_frames=num_frames, sample_every=sample_every)
    frames = [f for f, _ in frames_with_t]
    frame_times = [t for _, t in frames_with_t]
    frame_detections = [detect_vehicles(f, conf=conf) for f in frames]
    tracks = track_across_frames(frame_detections)

    results = []
    for track in tracks:
        speed_mps = estimate_speed_mps(track, frame_times, meters_per_pixel)
        if speed_mps is None:
            continue
        speed_mph = speed_mps * MPS_TO_MPH
        speeding, over_mph = flag_speeding(speed_mph, limit_mph)
        last = track[-1]
        results.append({
            "cls_name": last["cls_name"], "speed_mph": speed_mph, "speed_kmh": speed_mps * MPS_TO_KMH,
            "speeding": speeding, "over_mph": over_mph, "last_box": last["box"],
            "num_points": len(track),
        })
    return {"frames": frames, "frame_times": frame_times, "results": results}


def annotate_speeds(frame: np.ndarray, results: list[dict]) -> np.ndarray:
    out = frame.copy()
    for r in results:
        x1, y1, x2, y2 = (int(v) for v in r["last_box"])
        color = (0, 0, 255) if r["speeding"] else (0, 200, 0)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{r['speed_mph']:.0f} mph" + (" SPEEDING" if r["speeding"] else "")
        cv2.putText(out, label, (x1, max(15, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
    return out


def main():
    parser = argparse.ArgumentParser(description="Estimate vehicle speeds from a live HLS camera stream.")
    parser.add_argument("stream_url", help="HLS (.m3u8) stream URL")
    parser.add_argument("--lane-width-px", type=float, required=True,
                         help="Width of one traffic lane in pixels, measured on the frame, for calibration")
    parser.add_argument("--lane-width-m", type=float, default=US_LANE_WIDTH_M)
    parser.add_argument("--limit-mph", type=float, default=65.0)
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--save-annotated", default=None)
    args = parser.parse_args()

    meters_per_pixel = args.lane_width_m / args.lane_width_px
    result = check_speeds(args.stream_url, meters_per_pixel, args.limit_mph, num_frames=args.num_frames)

    print(f"Captured {len(result['frames'])} frames over {result['frame_times'][-1]:.1f}s")
    print(f"Tracked {len(result['results'])} vehicle(s):\n")
    for r in result["results"]:
        flag = " *** SPEEDING ***" if r["speeding"] else ""
        print(f"  {r['cls_name']:<10s} {r['speed_mph']:5.1f} mph ({r['speed_kmh']:5.1f} km/h)"
              f"  [{r['num_points']} tracked points]{flag}")

    if args.save_annotated and result["results"]:
        annotated = annotate_speeds(result["frames"][-1], result["results"])
        cv2.imwrite(args.save_annotated, annotated)
        print(f"\nSaved annotated frame to {args.save_annotated}")


if __name__ == "__main__":
    main()
