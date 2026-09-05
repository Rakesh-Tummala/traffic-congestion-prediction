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


def analyze_frame(frame: np.ndarray, conf: float = 0.25):
    """Run detection on a frame; return vehicle count, density ratio, and per-class breakdown."""
    model = get_model()
    results = model.predict(frame, conf=conf, classes=list(VEHICLE_CLASS_IDS.keys()), verbose=False)[0]

    h, w = frame.shape[:2]
    frame_area = h * w
    boxes_area = 0.0
    counts = {name: 0 for name in VEHICLE_CLASS_IDS.values()}

    for box in results.boxes:
        cls_id = int(box.cls.item())
        name = VEHICLE_CLASS_IDS.get(cls_id)
        if name is None:
            continue
        counts[name] += 1
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        boxes_area += max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))

    total_vehicles = sum(counts.values())
    density_ratio = min(1.0, boxes_area / frame_area) if frame_area else 0.0

    return {
        "vehicle_count": total_vehicles,
        "density_ratio": round(density_ratio, 4),
        "counts_by_class": counts,
        "annotated_frame": results.plot(),
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
    parser.add_argument("--conf", type=float, default=0.25)
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
