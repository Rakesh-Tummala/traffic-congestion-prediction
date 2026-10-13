"""Track vehicles across video frames and estimate each one's speed.

Pure logic with no model or video dependency, so it is unit-testable with synthetic
detections. `live_monitor.py` feeds it detections from YOLO on a live stream.

Tracking is nearest-neighbour with constant-velocity prediction: each existing track
predicts where its vehicle should be at the new frame's time, and detections are
matched to predictions greedily by distance (gated by the vehicle's own size). That is
deliberately simple and explainable, not a full multi-object tracker (no appearance
model, no re-identification), so it can mix up vehicles in dense stop-and-go traffic.

Speed needs a real-world scale and these cameras publish no calibration. Instead of
asking for a lane measurement, a flat-road pinhole-camera model is fitted from the
vehicles themselves:

* For a camera looking along a flat road, a vehicle's ground-contact row y and its pixel
  width w are linearly related: y = y_horizon + a * w, where a = camera_height / vehicle
  width. Fitting that line over many cars gives the horizon row and the camera height
  (taking a typical car to be CAR_WIDTH_M wide).
* The focal length is not recoverable from a single view, so it is assumed from a typical
  horizontal field of view (ASSUMED_HFOV_DEG). This is the largest uncertainty: a camera
  zoomed in or out from that assumption scales every speed by the same factor.
* Each vehicle's ground point is then converted to metres down the road (Z) and sideways
  (X), and its speed is the slope of that motion over the last couple of seconds.

If the fit is not convincing (too few cars, no clear trend, an implausible camera height,
e.g. a side-on view) calibration fails and speeds are NOT estimated, rather than guessed.
Treat the speeds as estimates (see README for the limits).
"""
import math
from dataclasses import dataclass, field

import numpy as np

MPS_TO_MPH = 2.2369362920544
CAR_WIDTH_M = 1.8
ASSUMED_HFOV_DEG = 62.0
MIN_CAMERA_HEIGHT_M, MAX_CAMERA_HEIGHT_M = 3.0, 25.0


@dataclass
class GroundCalibration:
    horizon_y: float      # image row of the horizon
    camera_height_m: float
    focal_px: float       # assumed, from ASSUMED_HFOV_DEG
    image_width: int
    correlation: float    # how well y vs width fit a line (diagnostic)
    n_samples: int

    def to_ground(self, x_center: float, y_bottom: float) -> tuple[float, float] | None:
        """(X sideways, Z forward) in metres for a ground point, or None at/above the horizon."""
        dy = y_bottom - self.horizon_y
        if dy <= 1.0:
            return None
        z = self.focal_px * self.camera_height_m / dy
        x = (x_center - self.image_width / 2) * z / self.focal_px
        return x, z


class CalibrationFitter:
    """Collects (pixel width, ground row) samples from confident cars, then fits the model."""

    def __init__(self, image_width: int, image_height: int, hfov_deg: float = ASSUMED_HFOV_DEG):
        self.image_width, self.image_height, self.hfov_deg = image_width, image_height, hfov_deg
        self.samples: list[tuple[float, float]] = []

    def add_detections(self, detections: list[dict], min_conf: float = 0.5) -> None:
        for d in detections:
            if d["cls_name"] == "car" and d.get("conf", 0) >= min_conf:
                x1, y1, x2, y2 = d["box"]
                if x2 - x1 >= 8:
                    self.samples.append((x2 - x1, y2))

    def fit(self, min_samples: int = 60, min_correlation: float = 0.7) -> GroundCalibration | None:
        if len(self.samples) < min_samples:
            return None
        w = np.array([s[0] for s in self.samples])
        y = np.array([s[1] for s in self.samples])
        if np.ptp(w) < 10:  # all cars at the same distance: slope is undetermined
            return None
        keep = np.ones(len(w), dtype=bool)
        for _ in range(3):  # iterative outlier rejection
            a, y_h = np.polyfit(w[keep], y[keep], 1)
            resid = y - (y_h + a * w)
            sigma = resid[keep].std() or 1.0
            keep = np.abs(resid) < 2.5 * sigma
            if keep.sum() < min_samples // 2:
                return None
        a, y_h = np.polyfit(w[keep], y[keep], 1)
        corr = float(np.corrcoef(w[keep], y[keep])[0, 1])
        height = a * CAR_WIDTH_M
        if (corr < min_correlation or not (MIN_CAMERA_HEIGHT_M <= height <= MAX_CAMERA_HEIGHT_M)
                or not (-0.8 * self.image_height <= y_h <= 0.8 * self.image_height)):
            return None
        focal = (self.image_width / 2) / math.tan(math.radians(self.hfov_deg / 2))
        return GroundCalibration(float(y_h), float(height), float(focal), self.image_width, corr, int(keep.sum()))


@dataclass
class Track:
    track_id: int
    cls_name: str
    history: list = field(default_factory=list)  # (t, x_center, y_bottom, width, height)
    last_box: tuple = (0, 0, 0, 0)
    last_conf: float = 0.0
    missed: int = 0
    flagged: bool = False
    class_votes: dict = field(default_factory=dict)

    def add(self, t: float, det: dict) -> None:
        x1, y1, x2, y2 = det["box"]
        self.history.append((t, (x1 + x2) / 2, y2, max(1.0, x2 - x1), max(1.0, y2 - y1)))
        self.last_box = (x1, y1, x2, y2)
        self.last_conf = det.get("conf", 0.0)
        self.class_votes[det["cls_name"]] = self.class_votes.get(det["cls_name"], 0) + 1
        self.cls_name = max(self.class_votes, key=self.class_votes.get)
        self.missed = 0


class CentroidTracker:
    def __init__(self, max_missed: int = 3, gate_factor: float = 1.6, max_history_s: float = 8.0):
        self.max_missed = max_missed
        self.gate_factor = gate_factor
        self.max_history_s = max_history_s
        self.tracks: list[Track] = []
        self._next_id = 1

    @staticmethod
    def _predict(track: Track, t: float) -> tuple[float, float]:
        h = track.history
        if len(h) >= 2:
            (t0, x0, y0, _, _), (t1, x1, y1, _, _) = h[-2], h[-1]
            dt = t1 - t0
            if dt > 1e-6:
                vx, vy = (x1 - x0) / dt, (y1 - y0) / dt
                return x1 + vx * (t - t1), y1 + vy * (t - t1)
        return h[-1][1], h[-1][2]

    def update(self, t: float, detections: list[dict]) -> list[Track]:
        """Match this frame's detections to tracks; returns the tracks seen in this frame."""
        pairs = []
        for ti, track in enumerate(self.tracks):
            px, py = self._predict(track, t)
            gate = self.gate_factor * max(track.history[-1][3], track.history[-1][4] * 0.8)
            for di, det in enumerate(detections):
                x1, y1, x2, y2 = det["box"]
                dist = float(np.hypot((x1 + x2) / 2 - px, y2 - py))
                if dist <= gate:
                    pairs.append((dist / gate, ti, di))
        pairs.sort()
        used_t, used_d, seen = set(), set(), []
        for _, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            self.tracks[ti].add(t, detections[di])
            seen.append(self.tracks[ti])
        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.missed += 1
        for di, det in enumerate(detections):
            if di not in used_d:
                track = Track(self._next_id, det["cls_name"])
                self._next_id += 1
                track.add(t, det)
                self.tracks.append(track)
                seen.append(track)
        self.tracks = [tr for tr in self.tracks if tr.missed <= self.max_missed]
        for tr in self.tracks:  # bound memory
            while len(tr.history) > 2 and tr.history[-1][0] - tr.history[0][0] > self.max_history_s:
                tr.history.pop(0)
        return seen


def estimate_speed_mph(track: Track, calib: GroundCalibration, window_s: float = 2.5,
                       min_span_s: float = 1.0, min_points: int = 4) -> float | None:
    """Speed from the slope of the track's recent ground-plane motion, or None if the track
    has too little history. Uses a least-squares slope (not just the endpoints) so one noisy
    box does not swing the result."""
    h = track.history
    if len(h) < min_points:
        return None
    t_end = h[-1][0]
    window = [p for p in h if t_end - p[0] <= window_s]
    if len(window) < min_points or window[-1][0] - window[0][0] < min_span_s:
        return None
    pts = [(p[0], calib.to_ground(p[1], p[2])) for p in window]
    pts = [(t, g) for t, g in pts if g is not None]
    if len(pts) < min_points:
        return None
    t = np.array([p[0] for p in pts])
    vx = np.polyfit(t, [p[1][0] for p in pts], 1)[0]
    vz = np.polyfit(t, [p[1][1] for p in pts], 1)[0]
    return float(math.hypot(vx, vz) * MPS_TO_MPH)
