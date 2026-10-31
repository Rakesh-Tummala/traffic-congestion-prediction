"""Diagnose the speed estimates on a recorded clip: distribution of the estimated speed of every tracked vehicle.

A freeway moving at a typical speed has a narrow speed distribution (a few mph either side of the median). A long
tail of very fast vehicles means tracking or geometry error, not speeders.

    python research/speed_diag.py research/data/clips/i5_crescent_*.mp4
"""
import glob
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from live_monitor import LiveMonitor  # noqa: E402


def main():
    for path in sorted({p for pat in sys.argv[1:] for p in glob.glob(pat)}):
        mon = LiveMonitor(path, os.path.basename(path), run_seconds=3600, detect_violations=False, log_challans=False)
        mon.start()
        speeds: dict[int, list[float]] = {}
        scale = calib = None
        while mon.running:
            time.sleep(0.05)
            snap = mon.snapshot()
            scale = snap["scale"] or scale
            calib = snap["calibration"] or calib
            for tid, v in snap["track_speeds"].items():
                speeds.setdefault(tid, []).append(v)
        per_track = np.array([np.median(v) for v in speeds.values() if len(v) >= 3])
        print(f"\n{os.path.basename(path)}: scale {scale}, camera fit {calib}, {len(per_track)} vehicles with speeds")
        if len(per_track):
            q = np.percentile(per_track, [5, 25, 50, 75, 95])
            print("  speed percentiles 5/25/50/75/95:", np.round(q, 1))
            for thr in (70, 80, 90):
                print(f"  share above {thr} mph: {np.mean(per_track > thr):.1%}")


if __name__ == "__main__":
    main()
