"""Check that everything the live demo depends on is working, and pick a camera with traffic.

    python demo_check.py            # about a minute

Prints PASS / FAIL per dependency (Caltrans cameras, a video stream, the models, OpenStreetMap
routing / place search / weather / speed limits), the current California time with a verdict on
whether it is a good time for the live parts, and the busiest cameras with a working video stream
right now, so you can open the one most likely to show cars and speeds.
"""
import os
import sys
import time

import cv2
import requests

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "src"))

from camera_time import camera_now  # noqa: E402
from live_cameras import fetch_cameras  # noqa: E402
from route_map import NOMINATIM_URL, USER_AGENT  # noqa: E402

results = []


def check(name, fn):
    t0 = time.time()
    try:
        detail = fn() or ""
        results.append((name, True, detail, time.time() - t0))
    except Exception as e:  # report every failure, never crash the checker
        results.append((name, False, str(e)[:110], time.time() - t0))
    ok, detail = results[-1][1], results[-1][2]
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<34}{detail}")


def main():
    print("Demo readiness check\n")
    now = camera_now()
    hour = now.hour
    if 6 <= hour < 10 or 15 <= hour < 19:
        verdict = "GOOD: commute hours, expect plenty of cars"
    elif 10 <= hour < 15:
        verdict = "OK: daytime, moderate traffic"
    elif 19 <= hour < 21 or 5 <= hour < 6:
        verdict = "DIM: dusk/dawn, fewer cars and poor light"
    else:
        verdict = "POOR: night in California, roads are nearly empty and dark"
    print(f"California time now: {now:%A %H:%M}  ->  {verdict}\n")

    print("Dependencies")
    cams = []

    def cameras():
        nonlocal cams
        for d in (7, 12):
            cams += fetch_cameras(d)
        return f"{len(cams)} cameras (districts 7 and 12)"

    check("Caltrans camera list", cameras)
    check("Trained models present", lambda: (_ for _ in ()).throw(FileNotFoundError("run python src/evaluate.py"))
          if not os.path.exists(os.path.join(HERE, "models", "lstm_traffic.pt")) and not
          [f for f in os.listdir(os.path.join(HERE, "models")) if f.endswith((".pt", ".joblib"))] else "found")
    check("OSRM road routing", lambda: requests.get(
        "https://router.project-osrm.org/route/v1/driving/-118.3,34.2;-118.5,34.0",
        params={"overview": "false"}, timeout=12).json()["code"])
    check("Nominatim place search", lambda: f"{len(requests.get(NOMINATIM_URL, params={'q': 'Burbank Airport', 'format': 'json', 'limit': 1}, headers={'User-Agent': USER_AGENT}, timeout=12).json())} result")
    check("Open-Meteo weather", lambda: f"{requests.get('https://api.open-meteo.com/v1/forecast', params={'latitude': 34.1, 'longitude': -118.3, 'current': 'temperature_2m'}, timeout=12).json()['current']['temperature_2m']} C")

    def overpass():
        from speed_limits import lookup_speed_limit
        got = lookup_speed_limit(34.1535, -118.3465, "I-5", timeout=12)
        return f"{got['limit_mph']} mph ({got['source']})" if got else "unreachable (the app falls back to an assumed 65 mph)"

    check("OpenStreetMap speed limit", overpass)

    print("\nBusiest freeway cameras with a working video stream right now")
    candidates = [c for c in cams if c["stream_url"] and c["name"].startswith(("I-5", "I-405", "I-10", "US-101", "I-110", "I-210"))]
    import random
    random.seed(int(time.time() // 600))  # a different sample every 10 minutes
    random.shuffle(candidates)
    from cv_module import analyze_frame, load_frame, CameraUnavailableError
    scored = []
    for c in candidates[:14]:
        try:
            if requests.get(c["stream_url"], timeout=6).status_code != 200:
                continue
            res = analyze_frame(load_frame(c["image_url"]))
            scored.append((res["vehicle_count"], c))
        except CameraUnavailableError:
            continue
        except Exception:
            continue
    scored.sort(key=lambda s: -s[0])
    if not scored:
        print("  (none responded: check your internet connection)")
    for n, c in scored[:5]:
        print(f"  {n:>3} vehicles   {c['name']} - {c['nearby_place']}   [District {c['district']}]")
    if scored and scored[0][0] < 8:
        print("\n  Few vehicles anywhere right now. Demo the prediction, grid and route tabs, and show the")
        print("  live monitor's calibration message; the speed/challan part needs traffic.")

    failed = [r for r in results if not r[1]]
    print(f"\n{len(results) - len(failed)}/{len(results)} dependency checks passed.")
    if failed:
        print("Failed: " + ", ".join(r[0] for r in failed) + ". Have screenshots ready for those parts.")


if __name__ == "__main__":
    main()
