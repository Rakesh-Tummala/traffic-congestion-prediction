"""Tests for the rule-based violation detectors, driven by synthetic tracks (no video, no model)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from vehicle_tracking import Track  # noqa: E402
from violations import ViolationDetector  # noqa: E402

W, H, DT = 640, 360, 0.27  # frame size and the ~3.7 Hz sample interval of the monitor


def box(x, y_bottom, w=30, h=22):
    return {"cls_name": "car", "conf": 0.9, "box": (x - w / 2, y_bottom - h, x + w / 2, y_bottom)}


class Car:
    """A synthetic vehicle: position as a function of time, tracked frame by frame."""

    def __init__(self, tid, pos_fn, t_start, t_end):
        self.track = Track(tid, "car")
        self.pos_fn, self.t_start, self.t_end = pos_fn, t_start, t_end

    def step(self, t):
        if not (self.t_start <= t <= self.t_end):
            return None
        x, y = self.pos_fn(t - self.t_start)
        self.track.add(t, box(x, y))
        return self.track


def run(cars, seconds, detector, ground_fn=None, scale=None, t_from=0.0):
    events = []
    t = t_from
    while t < seconds:
        tracks = [tr for tr in (c.step(t) for c in cars) if tr is not None]
        ground = ground_fn(t, cars) if ground_fn else None
        events += detector.update(t, tracks, ground, scale)
        t += DT
    return events


def stream(x, speed, y0, n, spacing, id0, direction=1):
    """n cars in one column, one every `spacing` seconds, moving down (+1) or up (-1) the image."""
    return [Car(id0 + i, lambda dt, x=x, y0=y0: (x, y0 + direction * speed * dt), i * spacing, i * spacing + 4.5) for i in range(n)]


def test_flow_field_learns_and_flags_wrong_way_only():
    cars = stream(100, 60, 50, 40, 1.0, 1)                              # left carriageway: down the image
    cars += stream(520, 60, 340, 40, 1.0, 100, direction=-1)             # right carriageway: up the image (legitimate)
    cars.append(Car(500, lambda dt: (100, 330 - 60 * dt), 31.0, 35.5))   # a car going UP the left carriageway
    d = ViolationDetector(W, H, learn_seconds=25.0)
    ev = run(cars, 42, d)
    kinds = [(e.kind, e.track_id) for e in ev]
    assert ("wrong_way", 500) in kinds
    assert all(tid == 500 for k, tid in kinds if k == "wrong_way")       # nobody on the opposite carriageway


def test_no_wrong_way_when_everyone_goes_the_same_way():
    d = ViolationDetector(W, H)
    ev = run(stream(100, 60, 50, 45, 1.0, 1), 45, d)
    assert not [e for e in ev if e.kind == "wrong_way"]


def test_wrong_way_is_not_flagged_before_the_flow_is_learned():
    cars = [Car(1, lambda dt: (100, 300 - 60 * dt), 2.0, 6.0)]
    d = ViolationDetector(W, H, learn_seconds=25.0)
    assert not run(cars, 12, d)


def test_cell_with_mixed_flow_is_never_used_for_wrong_way():
    cars = stream(100, 60, 50, 24, 1.0, 1) + stream(100, 60, 340, 24, 1.0, 100, direction=-1)   # both ways, same cell
    cars.append(Car(500, lambda dt: (100, 330 - 60 * dt), 31.0, 35.5))
    d = ViolationDetector(W, H, learn_seconds=25.0)
    assert not [e for e in run(cars, 42, d) if e.kind == "wrong_way"]


def test_stopped_vehicle_among_moving_traffic():
    cars = stream(100, 60, 50, 60, 0.8, 1) + stream(160, 60, 50, 60, 0.8, 200)
    cars.append(Car(900, lambda dt: (130 + (dt % 0.5) * 0.5, 200), 3.0, 40.0))   # parked in the lane, jitter < 1 px
    d = ViolationDetector(W, H, learn_seconds=10.0)
    ev = run(cars, 40, d)
    stopped = [e for e in ev if e.kind == "stopped_vehicle"]
    assert stopped and stopped[0].track_id == 900
    assert stopped[0].t - 3.0 >= 6.0


def test_a_queue_where_everything_is_stopped_is_not_flagged():
    cars = [Car(i, lambda dt, i=i: (100 + 40 * (i % 3), 150 + 30 * (i // 3)), 0.0, 40.0) for i in range(9)]
    d = ViolationDetector(W, H, learn_seconds=10.0)
    assert not [e for e in run(cars, 40, d) if e.kind == "stopped_vehicle"]


def test_a_moving_car_never_triggers_a_stop():
    cars = stream(100, 60, 50, 60, 0.8, 1)
    d = ViolationDetector(W, H, learn_seconds=10.0)
    assert not [e for e in run(cars, 40, d) if e.kind == "stopped_vehicle"]


# ---------------------------------------------------------------- lanes (ground plane) -------------
def lane_scenario(follower_gap_m, speed_mps=25.0, seconds=45.0):
    """Many cars on a straight road teach the road axis, then a follower trails a leader at a given gap."""
    cars, plan = [], {}

    def add(cid, lane_x, z0, v, t0, t1):
        c = Car(cid, lambda dt: (0, 0), t0, t1)
        plan[cid] = (lane_x, z0, v, t0)
        cars.append(c)

    for i in range(40):                       # background traffic in lane +3.5 m
        add(i + 1, 3.5, 10.0, 25.0, i * 0.8, i * 0.8 + 3.0)
    add(100, 0.0, 25.0 + follower_gap_m + 4.5, speed_mps, 30.0, 40.0)   # leader (further down the road)
    add(101, 0.0, 25.0, speed_mps, 30.0, 40.0)                          # follower
    return cars, plan


def ground_for(plan):
    def fn(t, cars):
        out = {}
        for c in cars:
            if c.t_start <= t <= c.t_end:
                x, z0, v, t0 = plan[c.track.track_id]
                out[c.track.track_id] = (x, z0 + v * (t - t0), v * 2.2369362920544)
        return out
    return fn


def drive(follower_gap_m):
    cars, plan = lane_scenario(follower_gap_m)
    for c in cars:                            # the image position only needs to exist; lane rules use the ground plane
        c.pos_fn = lambda dt: (300, 200 + dt)
    d = ViolationDetector(W, H, learn_seconds=25.0)
    return run(cars, 45, d, ground_fn=ground_for(plan), scale=1.0)


def test_tailgating_is_flagged_for_a_tiny_time_headway():
    ev = [e for e in drive(follower_gap_m=6.0) if e.kind == "tailgating"]    # 6 m gap at 25 m/s = 0.24 s
    assert ev and ev[0].track_id == 101
    assert ev[0].meta["headway"] < 0.6


def test_a_safe_following_distance_is_not_tailgating():
    assert not [e for e in drive(follower_gap_m=50.0) if e.kind == "tailgating"]   # 2 s headway


def test_headway_does_not_depend_on_the_camera_scale_factor():
    cars, plan = lane_scenario(follower_gap_m=6.0)
    for c in cars:
        c.pos_fn = lambda dt: (300, 200 + dt)
    # a wrong zoom makes every raw distance and raw speed too small by the same factor
    k = 0.5
    def fn(t, cs):
        g = ground_for(plan)(t, cs)
        return {i: (x * k, z * k, mph * k) for i, (x, z, mph) in g.items()}
    d = ViolationDetector(W, H, learn_seconds=25.0)
    ev = run(cars, 45, d, ground_fn=fn, scale=1 / k)
    assert [e for e in ev if e.kind == "tailgating"]


def test_weaving_is_flagged_but_a_single_lane_change_is_not():
    def make(two_changes):
        cars, plan = [], {}
        for i in range(36):          # background traffic that has left the scene before the test car moves
            c = Car(i + 1, lambda dt: (300, 200 + dt), i * 0.7, i * 0.7 + 3.0)
            plan[i + 1] = lambda tt: (3.5, 10.0 + 25.0 * tt)
            cars.append(c)
        c = Car(100, lambda dt: (300, 200 + dt), 30.0, 40.0)

        def path(tt):  # forward 25 m/s; lateral +3.5 m between 1-3 s, and (optionally) back between 5-7 s
            lat = 0.0 + 3.5 * min(1.0, max(0.0, (tt - 1.0) / 2.0))
            if two_changes:
                lat -= 3.5 * min(1.0, max(0.0, (tt - 5.0) / 2.0))
            return lat, 25.0 + 25.0 * tt
        plan[100] = path
        cars.append(c)
        return cars, plan

    def ground(plan):
        def fn(t, cs):
            return {c.track.track_id: (*plan[c.track.track_id](t - c.t_start), 56.0) for c in cs if c.t_start <= t <= c.t_end}
        return fn

    for two, expect in ((False, False), (True, True)):
        cars, plan = make(two)
        d = ViolationDetector(W, H, learn_seconds=25.0)
        ev = [e for e in run(cars, 42, d, ground_fn=ground(plan), scale=1.0) if e.kind == "unsafe_lane_change"]
        assert bool(ev) == expect, (two, ev)


def test_each_vehicle_is_reported_once_per_violation_type():
    cars = stream(100, 60, 50, 40, 1.0, 1) + [Car(500, lambda dt: (100, 330 - 60 * dt), 31.0, 38.0)]
    d = ViolationDetector(W, H, learn_seconds=25.0)
    ev = [e for e in run(cars, 45, d) if e.kind == "wrong_way" and e.track_id == 500]
    assert len(ev) == 1


def test_a_cell_that_also_carries_oncoming_traffic_is_never_used_to_accuse():
    """Regression for a real-traffic failure: oncoming cars were flagged because the flow of a cell that spans
    both carriageways had been 'learned' from a few early vehicles going one way."""
    cars = stream(100, 60, 50, 22, 1.0, 1)                       # 22 cars down in the first 22 s, nothing the other way
    cars += stream(100, 60, 340, 22, 1.0, 100, direction=-1)      # then a steady stream of oncoming cars
    for c in cars[22:]:
        c.t_start += 24.0
        c.t_end += 24.0
    d = ViolationDetector(W, H, learn_seconds=25.0)
    ev = [e for e in run(cars, 60, d) if e.kind == "wrong_way"]
    assert len(ev) <= 1       # at most the very first oncoming car, before anyone can tell it from a wrong-way car


def test_wrong_way_needs_a_well_supported_main_flow():
    cars = stream(100, 60, 50, 6, 1.0, 1)                         # only 6 distinct vehicles ever seen in the cell
    cars.append(Car(500, lambda dt: (100, 330 - 60 * dt), 31.0, 35.5))
    d = ViolationDetector(W, H, learn_seconds=25.0)
    assert not [e for e in run(cars, 42, d) if e.kind == "wrong_way"]


def test_a_second_wrong_way_car_later_in_the_session_is_still_flagged():
    cars = stream(100, 60, 50, 150, 1.0, 1)                               # steady one-way traffic for 150 s
    cars.append(Car(500, lambda dt: (100, 330 - 60 * dt), 31.0, 35.5))     # first wrong-way car
    cars.append(Car(501, lambda dt: (100, 330 - 60 * dt), 120.0, 124.5))   # another, 90 s later
    d = ViolationDetector(W, H, learn_seconds=25.0)
    flagged = {e.track_id for e in run(cars, 130, d) if e.kind == "wrong_way"}
    assert flagged == {500, 501}
