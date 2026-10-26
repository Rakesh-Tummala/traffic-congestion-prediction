"""Rule-based traffic-violation detectors that sit on top of the vehicle tracker.

Pure logic (no video, no model), so every rule is unit-testable with synthetic tracks.
`live_monitor.py` feeds it the tracks it already has and turns the events into SIMULATED challans.

What it can and cannot do (be honest about this in any write-up):

* ``wrong_way``  - a vehicle moving against the dominant flow of the part of the image it is in.
  The dominant flow is *learned* from the traffic during the calibration period, cell by cell, so the
  opposite carriageway (which legitimately flows the other way) is a different cell and is not flagged.
  Cells where the flow is not clearly one-directional are never used.
* ``stopped_vehicle`` - a vehicle that stays put for several seconds in a lane where the other traffic
  is moving (a breakdown or illegal stop). A queue where everything is slow is not flagged.
* ``tailgating`` - a follower whose time headway to the car ahead in the same lane stays under a
  threshold. Time headway = gap / speed is unchanged by the unknown camera zoom (both scale together),
  which is why it is usable on uncalibrated public cameras. EXPERIMENTAL: needs good ground positions.
* ``unsafe_lane_change`` - weaving (two lane changes in a few seconds) or a cut-in with a very small gap.
  Lanes are not drawn on the image; lateral position is measured across the learned road direction.
  EXPERIMENTAL: lateral errors of a metre or more are normal at these resolutions.

Not possible with these cameras (so deliberately not implemented): turn-signal (indicator) violations,
red-light jumping, anything that needs to see a small light or a painted line.
"""
from dataclasses import dataclass, field
import math

import numpy as np

KINDS = ("wrong_way", "stopped_vehicle", "tailgating", "unsafe_lane_change")

GRID_COLS, GRID_ROWS = 6, 4
MOVE_PX_PER_W = 0.45          # image speed (in car widths per second) above which a track counts as moving
STILL_PX_PER_W = 0.12         # ... and below which it counts as (nearly) stationary
MIN_FLOW_SAMPLES = 30         # samples a cell needs before its flow direction is trusted
MIN_FLOW_COHERENCE = 0.80     # |mean unit vector|: 1 = all traffic in the cell goes the same way
WRONG_WAY_COS = -0.70         # cosine between the vehicle's motion and the cell's flow
WRONG_WAY_CONFIRM = 4         # consecutive frames opposing the flow
STOP_SECONDS = 6.0            # stationary this long among moving traffic
LANE_CHANGE_M = 2.4           # lateral shift (scaled metres) that counts as a lane change
LANE_CHANGE_WINDOW_S = 4.0
WEAVE_WINDOW_S = 10.0
CUT_IN_HEADWAY_S = 0.4
TAILGATE_HEADWAY_S = 0.6
TAILGATE_SECONDS = 2.0
TAILGATE_MIN_MPS = 8.0        # about 18 mph; slower traffic is not tailgating
SAME_LANE_M = 1.6
CAR_LENGTH_M = 4.5


@dataclass
class Event:
    kind: str
    track_id: int
    t: float
    detail: str
    confidence: str            # "moderate" or "experimental"
    meta: dict = field(default_factory=dict)


def image_motion(track, window_s: float = 1.5, min_points: int = 4):
    """(dx, dy, dt) of the bottom-centre over the recent window, or None with too little history."""
    h = track.history
    if len(h) < min_points:
        return None
    t_end = h[-1][0]
    w = [p for p in h if t_end - p[0] <= window_s]
    if len(w) < min_points or w[-1][0] - w[0][0] < 0.6:
        return None
    return w[-1][1] - w[0][1], w[-1][2] - w[0][2], w[-1][0] - w[0][0]


def _width(track) -> float:
    return max(8.0, track.history[-1][3])


class FlowField:
    """Per-cell dominant direction of travel in the image, learned from tracks."""

    def __init__(self, width: int, height: int):
        self.width, self.height = width, height
        self.sx = np.zeros((GRID_ROWS, GRID_COLS))
        self.sy = np.zeros((GRID_ROWS, GRID_COLS))
        self.n = np.zeros((GRID_ROWS, GRID_COLS))

    def cell(self, x: float, y: float) -> tuple[int, int]:
        c = min(GRID_COLS - 1, max(0, int(x / self.width * GRID_COLS)))
        r = min(GRID_ROWS - 1, max(0, int(y / self.height * GRID_ROWS)))
        return r, c

    def add(self, x: float, y: float, dx: float, dy: float) -> None:
        norm = math.hypot(dx, dy)
        if norm < 1e-6:
            return
        r, c = self.cell(x, y)
        self.sx[r, c] += dx / norm
        self.sy[r, c] += dy / norm
        self.n[r, c] += 1

    def dominant(self, x: float, y: float):
        """(ux, uy, coherence, n) of the cell, or None when it has too few samples."""
        r, c = self.cell(x, y)
        n = self.n[r, c]
        if n < MIN_FLOW_SAMPLES:
            return None
        mx, my = self.sx[r, c] / n, self.sy[r, c] / n
        coh = math.hypot(mx, my)
        if coh < 1e-6:
            return None
        return mx / coh, my / coh, coh, int(n)


class ViolationDetector:
    def __init__(self, width: int, height: int, learn_seconds: float = 25.0):
        self.flow = FlowField(width, height)
        self.learn_seconds = learn_seconds
        self.t0: float | None = None
        self.road_k: float | None = None            # lateral drift dX/dZ of the road axis
        self._k_samples: list[float] = []
        self._wrong: dict[int, int] = {}
        self._anchors: list[dict] = []              # near-stationary spots: {x, y, t_first, t_last}
        self._lane_hist: dict[int, list[tuple]] = {}   # id -> [(t, L_raw, Z_raw)]
        self._changes: dict[int, list[float]] = {}
        self._tail_since: dict[int, float] = {}
        self._done: set[tuple[str, int]] = set()    # (kind, track_id) already reported

    # ------------------------------------------------------------------ helpers
    @property
    def learning(self) -> bool:
        return self.road_k is None

    def _report(self, kind, trk, t, detail, conf, **meta):
        key = (kind, trk.track_id)
        if key in self._done:
            return None
        self._done.add(key)
        return Event(kind, trk.track_id, t, detail, conf, meta)

    # ------------------------------------------------------------------ main entry
    def update(self, t: float, tracks: list, ground: dict | None = None, scale: float | None = None) -> list[Event]:
        """`tracks`: Track objects seen this frame. `ground`: {track_id: (X, Z, raw_mph or None)} in the
        calibrated ground plane (raw metres). `scale`: the factor that turns raw speeds (and so raw
        distances) into real ones, or None when speeds are unreliable (lane rules are then off)."""
        if self.t0 is None:
            self.t0 = t
        ground = ground or {}
        events: list[Event] = []
        motion = {tr.track_id: image_motion(tr) for tr in tracks}

        # ---- learn the flow field, then freeze it
        if t - self.t0 < self.learn_seconds:
            for tr in tracks:
                m = motion[tr.track_id]
                if m and math.hypot(m[0], m[1]) / m[2] >= MOVE_PX_PER_W * _width(tr):
                    self.flow.add(tr.history[-1][1], tr.history[-1][2], m[0], m[1])
        self._record_lane_samples(t, tracks, ground)
        self._learn_road_axis(t, tracks, ground)

        moving = [tr for tr in tracks if motion[tr.track_id]
                  and math.hypot(motion[tr.track_id][0], motion[tr.track_id][1]) / motion[tr.track_id][2] >= MOVE_PX_PER_W * _width(tr)]

        if t - self.t0 >= self.learn_seconds:
            events += self._wrong_way(t, tracks, motion)
            events += self._stopped(t, tracks, motion, moving)
        if scale and self.road_k is not None:
            events += self._lanes(t, tracks, ground, scale)
        return [e for e in events if e]

    # ------------------------------------------------------------------ wrong way
    def _wrong_way(self, t, tracks, motion):
        out = []
        for tr in tracks:
            m = motion[tr.track_id]
            if not m or len(tr.history) < 6:
                self._wrong[tr.track_id] = 0
                continue
            dx, dy, dt = m
            speed = math.hypot(dx, dy) / dt
            dom = self.flow.dominant(tr.history[-1][1], tr.history[-1][2])
            if dom is None or dom[2] < MIN_FLOW_COHERENCE or speed < MOVE_PX_PER_W * _width(tr):
                self._wrong[tr.track_id] = 0
                continue
            cos = (dx * dom[0] + dy * dom[1]) / (math.hypot(dx, dy) + 1e-9)
            if cos <= WRONG_WAY_COS:
                self._wrong[tr.track_id] = self._wrong.get(tr.track_id, 0) + 1
                if self._wrong[tr.track_id] >= WRONG_WAY_CONFIRM:
                    out.append(self._report("wrong_way", tr, t,
                                            f"moving against the learned flow of this part of the road (cos {cos:+.2f})",
                                            "moderate", cos=round(cos, 2), coherence=round(dom[2], 2)))
            else:
                self._wrong[tr.track_id] = 0
        return out

    # ------------------------------------------------------------------ stopped vehicle
    def _stopped(self, t, tracks, motion, moving):
        out = []
        # forget spots nobody has been seen still at for a while
        self._anchors = [a for a in self._anchors if t - a["t_last"] <= 2.0]
        n_other_moving = len(moving)
        for tr in tracks:
            m = motion[tr.track_id]
            if not m:
                continue
            speed = math.hypot(m[0], m[1]) / m[2]
            if speed >= STILL_PX_PER_W * _width(tr):
                continue                      # moving, or too uncertain: never creates or refreshes an anchor
            x, y = tr.history[-1][1], tr.history[-1][2]
            r = max(5.0, 0.3 * _width(tr))
            a = next((a for a in self._anchors if math.hypot(a["x"] - x, a["y"] - y) <= r), None)
            if a is None:
                self._anchors.append({"x": x, "y": y, "t_first": t, "t_last": t})
                continue
            a["t_last"] = t
            dom = self.flow.dominant(x, y)
            flowing_lane = dom is not None and dom[2] >= 0.6
            if t - a["t_first"] >= STOP_SECONDS and flowing_lane and n_other_moving >= 3 \
                    and n_other_moving >= 0.6 * max(1, len(tracks) - 1):
                out.append(self._report("stopped_vehicle", tr, t,
                                        f"stationary for {t - a['t_first']:.0f} s while {n_other_moving} other vehicles kept moving",
                                        "moderate", seconds=round(t - a["t_first"], 1)))
        return out

    # ------------------------------------------------------------------ lanes, lane changes, tailgating
    def _learn_road_axis(self, t, tracks, ground):
        if self.road_k is not None or t - self.t0 < self.learn_seconds:
            return
        # use the slopes gathered while learning; need a decent number of long tracks
        if len(self._k_samples) >= 15:
            self.road_k = float(np.median(self._k_samples))
        else:
            self.road_k = None

    def _record_lane_samples(self, t, tracks, ground):
        for tr in tracks:
            g = ground.get(tr.track_id)
            if g is None:
                continue
            X, Z = g[0], g[1]
            hist = self._lane_hist.setdefault(tr.track_id, [])
            hist.append((t, X, Z))
            while hist and t - hist[0][0] > 12.0:
                hist.pop(0)
            if self.road_k is None and len(hist) >= 6:  # learning phase: road direction from long, forward tracks
                dz = hist[-1][2] - hist[0][2]
                if abs(dz) >= 8.0:
                    self._k_samples.append((hist[-1][1] - hist[0][1]) / dz)
                    del hist[:3]

    def _lanes(self, t, tracks, ground, scale):
        if self.road_k is None:
            return []
        k = self.road_k
        out = []
        pos = {}
        for tr in tracks:
            g = ground.get(tr.track_id)
            if g is not None:
                pos[tr.track_id] = ((g[0] - k * g[1]) * scale, g[1] * scale, g[2])  # lateral m, forward m, raw mph
        by_id = {tr.track_id: tr for tr in tracks}
        # ---- lane changes and weaving
        for tid, hist in self._lane_hist.items():
            if tid not in by_id or len(hist) < 6:
                continue
            win = [(h[0], (h[1] - k * h[2]) * scale, h[2] * scale) for h in hist if t - h[0] <= LANE_CHANGE_WINDOW_S]
            if len(win) < 6 or win[-1][0] - win[0][0] < 1.5 or abs(win[-1][2] - win[0][2]) < 8.0:
                continue
            lat = np.array([w[1] for w in win])
            net = lat[-1] - lat[0]
            span = lat.max() - lat.min()
            recent = self._changes.setdefault(tid, [])
            if abs(net) >= LANE_CHANGE_M and abs(net) >= 0.8 * span and (not recent or t - recent[-1] > 3.0):
                recent.append(t)
                trk = by_id[tid]
                n_recent = sum(1 for c in recent if t - c <= WEAVE_WINDOW_S)
                if n_recent >= 2:
                    out.append(self._report("unsafe_lane_change", trk, t,
                                            f"weaving: {n_recent} lane changes within {WEAVE_WINDOW_S:.0f} s",
                                            "experimental", pattern="weaving"))
                else:
                    cut = self._cut_in(tid, pos, scale)
                    if cut:
                        out.append(self._report("unsafe_lane_change", trk, t, cut, "experimental", pattern="cut-in"))
        # ---- tailgating
        for tid, p in pos.items():
            raw_mph = p[2]
            if raw_mph is None:
                continue
            v = raw_mph * scale / 2.2369362920544  # m/s
            if v < TAILGATE_MIN_MPS:
                self._tail_since.pop(tid, None)
                continue
            direction = self._direction(tid)
            if direction == 0:
                continue
            leader = None
            for oid, q in pos.items():
                if oid == tid or abs(q[0] - p[0]) > SAME_LANE_M or self._direction(oid) != direction:
                    continue
                ahead = (q[1] - p[1]) * direction
                if ahead > 0 and (leader is None or ahead < leader[0]):
                    leader = (ahead, oid)
            if leader is None:
                self._tail_since.pop(tid, None)
                continue
            gap = max(0.1, leader[0] - CAR_LENGTH_M)
            headway = gap / v
            if headway < TAILGATE_HEADWAY_S:
                since = self._tail_since.setdefault(tid, t)
                if t - since >= TAILGATE_SECONDS:
                    out.append(self._report("tailgating", by_id[tid], t,
                                            f"time headway {headway:.2f} s for {t - since:.1f} s (gap about {gap:.0f} m)",
                                            "experimental", headway=round(headway, 2), seconds=round(t - since, 1)))
            else:
                self._tail_since.pop(tid, None)
        return out

    def _direction(self, tid: int) -> int:
        """+1 when the vehicle moves away from the camera (Z increasing), -1 towards it, 0 unknown."""
        h = self._lane_hist.get(tid)
        if not h or len(h) < 4:
            return 0
        dz = h[-1][2] - h[0][2]
        return 0 if abs(dz) < 2.0 else (1 if dz > 0 else -1)

    def _cut_in(self, tid, pos, scale):
        me = pos.get(tid)
        if me is None:
            return None
        for oid, q in pos.items():
            if oid == tid or abs(q[0] - me[0]) > SAME_LANE_M:
                continue
            gap = abs(q[1] - me[1]) - CAR_LENGTH_M
            speed_mps = max(6.0, (me[2] or 0) * scale * 0.44704)  # floor so a near-zero speed cannot give a huge headway
            if 0 < gap and gap / speed_mps < CUT_IN_HEADWAY_S:
                return f"changed lane into a gap of about {max(gap, 0):.0f} m"
        return None
