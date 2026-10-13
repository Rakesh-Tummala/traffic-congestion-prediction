"""Vehicle detection and density estimation from a camera frame using YOLOv8.

Works on a local image/video file, a live snapshot URL (e.g. a public DOT
traffic camera still), or a webcam index. Detects COCO vehicle classes
(car, motorcycle, bus, truck) and reports a count and an estimated
occupancy/density ratio (vehicle bounding-box area / frame area) which is
used downstream as a live feature for the regression / LSTM models.
"""
import argparse
import io
import os

import cv2
import numpy as np
import requests
from ultralytics import YOLO

VEHICLE_CLASS_IDS = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}
# yolov8n undercounts badly in dense/occluded traffic (verified: 9 vehicles found in a
# jam with 30+ actually present); yolov8s trades a bit of speed for meaningfully better
# recall on small/overlapping vehicles and is still fast enough for single-frame inference.
MODEL_WEIGHTS = "yolov8s.pt"  # pretrained COCO model, auto-downloaded on first use

# Detection confidence threshold. Chosen from a sweep on 124 hand-labelled live Caltrans frames
# (cv_eval/results/report.md): at the YOLO default of 0.25 the count was on average 7 vehicles
# low (MAE 7.4); 0.08 cuts that to MAE 4.6 and bias -2.5 without hurting near-empty frames.
# Picked among a handful of values on the same frames, so expect a little optimism.
DEFAULT_CONF = 0.08

_model = None


def get_model() -> YOLO:
    global _model
    if _model is None:
        _model = YOLO(MODEL_WEIGHTS)
    return _model


def load_frame(source: str) -> np.ndarray:
    """Load a single BGR frame from a file path, an http(s) URL, or a webcam index string."""
    if source.startswith("http://") or source.startswith("https://"):
        resp = requests.get(source, timeout=10)
        resp.raise_for_status()
        arr = np.frombuffer(resp.content, dtype=np.uint8)
        frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError(f"Could not decode image from URL: {source}")
        return frame

    if source.isdigit():
        cap = cv2.VideoCapture(int(source))
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise ValueError(f"Could not read from webcam index {source}")
        return frame

    frame = cv2.imread(source)
    if frame is not None:
        return frame

    cap = cv2.VideoCapture(source)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"Could not load frame from: {source}")
    return frame


class CameraUnavailableError(ValueError):
    """The image is Caltrans' "Temporarily Unavailable" placeholder, not a real camera view."""


# Measured on 162 live Caltrans frames (cv_eval/): the placeholder is 86% near-white
# pixels, the most washed-out real frame 22%. 0.6 sits well clear of both.
UNAVAILABLE_WHITE_FRACTION = 0.6


def is_unavailable_frame(frame: np.ndarray) -> bool:
    """True for the mostly-white "Temporarily Unavailable" image a Caltrans camera serves
    when its feed is down. About one in five sampled cameras was showing it, and with no
    vehicles in it a detector would report an empty road ("Low") instead of "no data"."""
    if frame is None or frame.size == 0:
        return False
    return float((frame.min(axis=2) >= 235).mean()) >= UNAVAILABLE_WHITE_FRACTION


def tile_windows(height: int, width: int, grid: tuple[int, int] = (2, 2), overlap: float = 0.2):
    """Overlapping (x0, y0, x1, y1) windows covering the frame, grid[0] columns by grid[1] rows."""
    cols, rows = grid
    tw, th = width / (cols - (cols - 1) * overlap), height / (rows - (rows - 1) * overlap)
    xs = [round(i * (width - tw) / max(1, cols - 1)) for i in range(cols)]
    ys = [round(j * (height - th) / max(1, rows - 1)) for j in range(rows)]
    return [(x, y, min(width, round(x + tw)), min(height, round(y + th))) for y in ys for x in xs]


def merge_boxes(dets: list[dict], ios_threshold: float = 0.6) -> list[dict]:
    """Drop duplicate detections of the same vehicle found by overlapping windows.

    Highest confidence first; a box is dropped if, for a kept box of the same class, the
    intersection covers more than `ios_threshold` of the SMALLER box (intersection over
    smaller, not IoU: a car cut by a tile edge is a small fragment of the whole car and has a
    low IoU with it, but is almost entirely inside it)."""
    kept: list[dict] = []
    for d in sorted(dets, key=lambda x: -x["conf"]):
        x1, y1, x2, y2 = d["box"]
        area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        duplicate = False
        for k in kept:
            if k["cls_id"] != d["cls_id"]:
                continue
            kx1, ky1, kx2, ky2 = k["box"]
            iw, ih = min(x2, kx2) - max(x1, kx1), min(y2, ky2) - max(y1, ky1)
            if iw <= 0 or ih <= 0:
                continue
            smaller = min(area, max(0.0, kx2 - kx1) * max(0.0, ky2 - ky1))
            if smaller > 0 and iw * ih / smaller > ios_threshold:
                duplicate = True
                break
        if not duplicate:
            kept.append(d)
    return kept


def detect_vehicles_raw(model, frame: np.ndarray, conf: float, imgsz: int = 640, tiled: bool = False,
                        grid: tuple[int, int] = (2, 2)) -> list[dict]:
    """Vehicle boxes as dicts {box, conf, cls_id}. With tiled=True the frame is also searched in
    overlapping windows (each upscaled to the model's input size), which finds small, distant
    vehicles the single full-frame pass misses; results are merged with merge_boxes()."""
    ids = list(VEHICLE_CLASS_IDS.keys())

    def run(img, dx=0, dy=0):
        res = model.predict(img, conf=conf, imgsz=imgsz, classes=ids, verbose=False)[0]
        out = []
        for b in res.boxes:
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            out.append({"box": (x1 + dx, y1 + dy, x2 + dx, y2 + dy), "conf": float(b.conf.item()),
                        "cls_id": int(b.cls.item())})
        return out

    dets = run(frame)
    if tiled:
        h, w = frame.shape[:2]
        for x0, y0, x1, y1 in tile_windows(h, w, grid):
            dets += run(frame[y0:y1, x0:x1], x0, y0)
        dets = merge_boxes(dets)
    return dets


def analyze_frame(frame: np.ndarray, conf: float = DEFAULT_CONF, tiled: bool = False):
    """Run detection on a frame; return vehicle count, density ratio, and per-class breakdown.
    `tiled=True` is the slower, more sensitive mode for distant vehicles (about 5 model passes).
    Raises CameraUnavailableError for a camera's "Temporarily Unavailable" placeholder."""
    if is_unavailable_frame(frame):
        raise CameraUnavailableError("This camera's feed is temporarily unavailable (Caltrans "
                                     "placeholder image), so there is nothing to detect.")
    model = get_model()
    h, w = frame.shape[:2]
    dets = detect_vehicles_raw(model, frame, conf, tiled=tiled)

    boxes_area = 0.0
    counts = {name: 0 for name in VEHICLE_CLASS_IDS.values()}
    annotated = frame.copy()
    for d in dets:
        name = VEHICLE_CLASS_IDS[d["cls_id"]]
        counts[name] += 1
        x1, y1, x2, y2 = d["box"]
        boxes_area += max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 144, 30), 2)
        cv2.putText(annotated, f"{name} {d['conf']:.2f}", (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 144, 30), 1)

    density_ratio = min(1.0, boxes_area / (h * w)) if h * w else 0.0
    return {
        "vehicle_count": sum(counts.values()),
        "density_ratio": round(density_ratio, 4),
        "counts_by_class": counts,
        "annotated_frame": annotated,
    }


CONGESTION_LEVELS = ["Low", "Moderate", "High", "Severe"]


def congestion_from_density(density_ratio: float) -> str:
    """Rough heuristic mapping image occupancy to a congestion label (for live display only;
    the trained regression/LSTM models produce the calibrated numeric prediction)."""
    if density_ratio < 0.05:
        return "Low"
    if density_ratio < 0.15:
        return "Moderate"
    if density_ratio < 0.30:
        return "High"
    return "Severe"


def main():
    parser = argparse.ArgumentParser(description="Count vehicles / estimate density from a camera frame.")
    parser.add_argument("source", help="Image path, video path, snapshot URL, or webcam index (e.g. 0)")
    parser.add_argument("--conf", type=float, default=DEFAULT_CONF)
    parser.add_argument("--save", default=None, help="Path to save the annotated frame")
    args = parser.parse_args()

    frame = load_frame(args.source)
    result = analyze_frame(frame, conf=args.conf)

    print(f"Vehicle count: {result['vehicle_count']}")
    print(f"Density ratio: {result['density_ratio']}")
    print(f"By class: {result['counts_by_class']}")
    print(f"Heuristic congestion (image-only): {congestion_from_density(result['density_ratio'])}")

    if args.save:
        cv2.imwrite(args.save, result["annotated_frame"])
        print(f"Saved annotated frame to {args.save}")


if __name__ == "__main__":
    main()
