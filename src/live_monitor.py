"""Run detection, tracking and (simulated) speed enforcement on a live Caltrans video stream.

A background thread reads the camera's HLS stream, runs YOLO on every few frames, tracks the
vehicles, estimates their speeds, and when a vehicle is clearly over the limit it grabs the
vehicle's picture, optionally reads the plate and identifies colour/type, and generates a
SIMULATED challan from those real observations. The Streamlit page only polls `snapshot()`.

Honest limits, enforced in code:
* Speed comes from a flat-road camera model fitted from the cars themselves
  (vehicle_tracking.py). If that fit is not convincing, speeds are switched off for the camera.
* The model has an assumed focal length, so absolute speeds are uncertain. Speeds are therefore
  also scaled so the median tracked vehicle matches `reference_mph` (the typical speed of
  traffic here), and if that needs a big correction the camera is marked unreliable and no
  challans are issued.
* Nothing here is a real citation: challans carry the simulation disclaimer. Evidence crops are
  held in memory only; the on-disk log has text fields only.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

from camera_time import camera_now
from cv_module import DEFAULT_CONF, VEHICLE_CLASS_IDS, congestion_from_density
from echallan import append_to_log, generate_challan
from vehicle_tracking import CalibrationFitter, CentroidTracker, estimate_speed_mph

SAMPLE_EVERY = 4              # process every 4th frame (15 fps stream -> ~3.75 Hz)
TRACK_CONF = 0.30             # confidence for detections fed to the tracker
CALIBRATION_SECONDS = 25.0    # stream seconds spent collecting cars for the camera fit
MIN_TRACKS_FOR_SCALE = 12     # tracked vehicles needed before the traffic-speed scale is trusted
SCALE_RANGE = (0.4, 2.5)      # a larger correction than this means the camera model is not usable
MAX_PLAUSIBLE_MPH = 100.0     # faster than this is a tracking glitch, never a violation
# A car cannot move far faster than the traffic around it. In a live test on a busy freeway, half the
# challans (97-119 mph) came from tracks that jumped between cars, while real ones clustered at 56-63 mph
# against typical traffic of 60. Speeds above this multiple of the typical traffic speed are treated as
# tracking errors: shown without a speed and never flagged.
MAX_SPEED_RATIO = 1.6
# Pixel noise in a far vehicle's position is magnified into a big speed error (a 3 px wobble is
# about a metre at 100 m), and very near vehicles leave the frame quickly. Live testing showed
# far cars reading far faster than near ones in the same traffic, so speeds and challans are
# limited to this band of distances from the camera (metres down the road).
RANGE_BAND_M = (8.0, 75.0)
CONFIRMATIONS = 3             # consecutive evaluations above the threshold before a challan
MAX_EVENTS = 12               # a session never issues more than this many challans
MIN_EVENT_GAP_S = 3.0         # stream seconds between challans (a tracker glitch can't machine-gun them)
IDLE_STOP_SECONDS = 45.0      # stop if the page stops polling (user closed it)

GREEN, ORANGE, RED, GRAY = (0, 200, 0), (0, 165, 255), (0, 0, 255), (160, 160, 160)


def violation_threshold(limit_mph: float, margin_mph: float) -> float:
    return limit_mph + margin_mph


def is_plausible_speed(mph: float, reference_mph: float) -> bool:
    """False for a speed that is almost certainly a tracking error: above the absolute cap, or far above
    the typical traffic speed that the camera's speeds were scaled to."""
    return mph <= MAX_PLAUSIBLE_MPH and mph <= MAX_SPEED_RATIO * reference_mph


def speed_scale(raw_speeds: list[float], reference_mph: float, min_tracks: int = MIN_TRACKS_FOR_SCALE):
    """Factor that makes the median tracked speed equal `reference_mph`, or None when there are
    too few tracks or the needed correction is outside SCALE_RANGE (model not usable here)."""
    if len(raw_speeds) < min_tracks:
        return None
    median = float(np.median(raw_speeds))
    if median <= 0:
        return None
    scale = reference_mph / median
    return scale if SCALE_RANGE[0] <= scale <= SCALE_RANGE[1] else None


class LiveMonitor:
    def __init__(self, stream_url: str, camera_name: str, limit_mph: float = 65.0, margin_mph: float = 15.0,
                 reference_mph: float = 60.0, run_seconds: float = 120.0, read_plates: bool = False,
                 identify: bool = False, weights: str = "yolov8s.pt", limit_source: str = ""):
        self.stream_url, self.camera_name, self.limit_source = stream_url, camera_name, limit_source
        self.limit_mph, self.margin_mph, self.reference_mph = limit_mph, margin_mph, reference_mph
        self.run_seconds, self.read_plates, self.identify, self.weights = run_seconds, read_plates, identify, weights
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._pool = ThreadPoolExecutor(max_workers=1)  # plate/appearance work must not stall the stream
        self._last_poll = time.time()
        self._started_at = 0.0
        self.state = {"status": "idle", "message": "", "frame": None, "vehicle_count": 0, "density": 0.0,
                      "congestion": None, "speed_status": "waiting", "events": [], "tracks_with_speed": 0,
                      "scale": None, "calibration": None, "elapsed": 0.0, "fps": 0.0}

    # ------------------------------------------------------------------ control
    def start(self) -> None:
        self._started_at = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True, name="live-monitor")
        self._set(status="starting", message="Opening the live stream...")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def snapshot(self) -> dict:
        self._last_poll = time.time()
        with self._lock:
            snap = dict(self.state)
            snap["events"] = list(self.state["events"])
            snap["elapsed"] = time.time() - self._started_at if self._started_at else 0.0
            if snap["frame"] is not None:
                snap["frame"] = snap["frame"].copy()
            return snap

    def _set(self, **kw) -> None:
        with self._lock:
            self.state.update(kw)

    # ------------------------------------------------------------------ worker
    def _should_stop(self) -> bool:
        return (self._stop.is_set() or time.time() - self._started_at > self.run_seconds
                or time.time() - self._last_poll > IDLE_STOP_SECONDS)

    def _run(self) -> None:
        cap = None
        try:
            from ultralytics import YOLO  # own instance: the app's shared model is not thread-safe

            cap = cv2.VideoCapture()
            cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 10000)
            cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 10000)
            if not cap.open(self.stream_url):
                self._set(status="error", message="Could not open this camera's video stream. Many Caltrans "
                                                  "streams are offline; pick another camera.")
                return
            model = YOLO(self.weights)
            tracker, fitter, calib, calib_state = CentroidTracker(), None, None, "collecting"
            raw_speed: dict[int, float] = {}
            consecutive: dict[int, int] = {}
            n, t_first, last_wall, density_hist, last_event_t = 0, None, time.time(), [], -1e9
            self._set(status="calibrating", message="Watching traffic to calibrate the camera view...")

            while not self._should_stop():
                if not cap.grab():
                    self._set(status="error" if n == 0 else "stopped",
                              message="The stream ended or stopped responding.")
                    return
                n += 1
                if n % SAMPLE_EVERY:
                    continue
                ok, frame = cap.retrieve()
                if not ok:
                    continue
                t = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0
                t_first = t if t_first is None else t_first
                h, w = frame.shape[:2]
                if fitter is None:
                    fitter = CalibrationFitter(w, h)

                res = model.predict(frame, conf=DEFAULT_CONF, imgsz=640, classes=list(VEHICLE_CLASS_IDS),
                                    verbose=False)[0]
                dets = [{"cls_name": VEHICLE_CLASS_IDS[int(b.cls.item())], "conf": float(b.conf.item()),
                         "box": tuple(b.xyxy[0].tolist())} for b in res.boxes]
                area = sum(max(0.0, d["box"][2] - d["box"][0]) * max(0.0, d["box"][3] - d["box"][1]) for d in dets)
                density_hist = (density_hist + [min(1.0, area / (w * h))])[-6:]
                tracked = tracker.update(t, [d for d in dets if d["conf"] >= TRACK_CONF])

                # --- camera calibration, once, after enough stream time
                if calib_state == "collecting":
                    fitter.add_detections(dets)
                    if t - t_first >= CALIBRATION_SECONDS:
                        calib = fitter.fit()
                        if calib is None:
                            calib_state = "failed"
                            self._set(speed_status="unreliable", status="monitoring",
                                      message="Couldn't calibrate this camera's geometry (it needs a clear view "
                                              "of cars at different distances along the road). Detection "
                                              "continues; speeds and challans are off for this camera.")
                        else:
                            calib_state = "ok"
                            self._set(calibration={"camera_height_m": round(calib.camera_height_m, 1),
                                                   "correlation": round(calib.correlation, 2)},
                                      status="monitoring", message="")

                scale = None
                speed_status = {"collecting": "calibrating", "failed": "unreliable"}.get(calib_state, "ok")
                if calib is not None:
                    for trk in tracked:
                        g = calib.to_ground((trk.last_box[0] + trk.last_box[2]) / 2, trk.last_box[3])
                        if g is None or not (RANGE_BAND_M[0] <= g[1] <= RANGE_BAND_M[1]):
                            raw_speed.pop(trk.track_id, None)  # outside the reliable band: no speed
                            continue
                        s = estimate_speed_mph(trk, calib)
                        if s is not None and s <= MAX_PLAUSIBLE_MPH * 3:
                            raw_speed[trk.track_id] = s
                    scale = speed_scale(list(raw_speed.values()), self.reference_mph)
                    if len(raw_speed) < MIN_TRACKS_FOR_SCALE:
                        speed_status = "calibrating"
                    elif scale is None:
                        speed_status = "unreliable"

                annotated = frame.copy()
                for trk in tracked:
                    x1, y1, x2, y2 = (int(v) for v in trk.last_box)
                    label, color = trk.cls_name, GRAY
                    if scale is not None and trk.track_id in raw_speed:
                        mph = raw_speed[trk.track_id] * scale
                        plausible = is_plausible_speed(mph, self.reference_mph)
                        label = f"{trk.cls_name} {mph:.0f} mph" if plausible else trk.cls_name
                        over = mph - self.limit_mph
                        color = (RED if over >= self.margin_mph else ORANGE if over > 0 else GREEN) if plausible else GRAY
                        if plausible and mph >= violation_threshold(self.limit_mph, self.margin_mph) \
                                and not trk.flagged and len(trk.history) >= 5:
                            consecutive[trk.track_id] = consecutive.get(trk.track_id, 0) + 1
                            if consecutive[trk.track_id] >= CONFIRMATIONS:  # several in a row, not one noisy one
                                trk.flagged = True
                                if len(self.state["events"]) < MAX_EVENTS and t - last_event_t >= MIN_EVENT_GAP_S:
                                    last_event_t = t
                                    self._issue(frame, trk, mph)
                                elif len(self.state["events"]) >= MAX_EVENTS and not self.state["message"]:
                                    self._set(message=f"Reached the limit of {MAX_EVENTS} challans for this session; "
                                                      "detection continues but no more are issued.")
                        else:
                            consecutive[trk.track_id] = 0
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated, label, (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                now = time.time()
                density = float(np.mean(density_hist))
                self._set(frame=annotated, vehicle_count=len(dets), density=round(density, 4),
                          congestion=congestion_from_density(density), speed_status=speed_status,
                          tracks_with_speed=len(raw_speed), scale=scale,
                          fps=1.0 / max(1e-3, now - last_wall))
                last_wall = now
            self._set(status="stopped", message="Monitoring finished.")
        except Exception as e:  # surface any failure on the page instead of dying silently
            self._set(status="error", message=f"Live monitor failed: {e}")
        finally:
            if cap is not None:
                cap.release()
            self._pool.shutdown(wait=False)

    # ------------------------------------------------------------------ violations
    def _issue(self, frame: np.ndarray, trk, mph: float) -> None:
        """Queue challan creation for one flagged vehicle (plate/appearance may take seconds)."""
        x1, y1, x2, y2 = (int(v) for v in trk.last_box)
        h, w = frame.shape[:2]
        pad_x, pad_y = int(0.15 * (x2 - x1)), int(0.15 * (y2 - y1))
        crop = frame[max(0, y1 - pad_y):min(h, y2 + pad_y), max(0, x1 - pad_x):min(w, x2 + pad_x)].copy()
        event = {"time": camera_now().strftime("%H:%M:%S"), "cls_name": trk.cls_name, "speed_mph": mph,
                 "crop": crop, "challan": None, "plate": None, "attrs": None, "status": "processing"}
        with self._lock:
            self.state["events"].append(event)
        self._pool.submit(self._finalize_event, event, frame.copy(), trk.last_box, trk.cls_name, mph)

    def _finalize_event(self, event: dict, frame: np.ndarray, box: tuple, cls_name: str, mph: float) -> None:
        try:
            plate_text, plate_conf, desc = "", 0.0, ""
            if self.read_plates:
                from anpr import read_plate
                plate = read_plate(frame, box)
                event["plate"] = plate
                plate_text, plate_conf = plate["text"] or "", plate["confidence"]
            if self.identify:
                from vehicle_attributes import describe_vehicle
                attrs = describe_vehicle(frame, box)
                event["attrs"] = attrs
                desc = " ".join(p for p in (attrs.get("color"), attrs.get("type")) if p)
                if cls_name in ("car", "truck"):
                    from vehicle_attributes import guess_make_model
                    mm = guess_make_model(frame, box)  # None fields unless the crop is big and CLIP is sure
                    event["make_model"] = mm
                    if mm["model"]:
                        desc += f", possibly a {mm['model']} ({mm['model_confidence']:.0%} confidence)"
                    elif mm["make"]:
                        desc += f", possibly a {mm['make']} ({mm['make_confidence']:.0%} confidence)"
                desc = f"{desc} (best guess)" if desc else ""
            challan = generate_challan(
                {"speeding": True, "cls_name": cls_name, "speed_mph": mph, "over_mph": mph - self.limit_mph},
                self.camera_name, self.limit_mph, vehicle_number=plate_text,
                plate_auto_detected=bool(plate_text), plate_confidence=plate_conf,
                vehicle_description=desc, source="live monitor", limit_source=self.limit_source)
            append_to_log(challan)
            event["challan"] = challan
            event["status"] = "done"
        except Exception as e:
            event["status"] = f"failed: {e}"
