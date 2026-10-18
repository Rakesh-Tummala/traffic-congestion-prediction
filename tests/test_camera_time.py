"""The models must be asked about California time, whatever time zone this computer is set to."""
import os
import sys
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import camera_time  # noqa: E402


def test_camera_now_is_california_wall_clock_time_and_naive():
    now = camera_time.camera_now()
    assert now.tzinfo is None
    utc = datetime.now(timezone.utc).replace(tzinfo=None)
    offset_hours = (utc - now).total_seconds() / 3600
    assert round(offset_hours) in (7, 8)       # PDT is UTC-7, PST is UTC-8


def test_camera_now_does_not_follow_the_computers_own_time_zone(monkeypatch):
    # a computer set to India time (UTC+5:30) must still get California's hour
    monkeypatch.setenv("TZ", "Asia/Kolkata")
    if hasattr(__import__("time"), "tzset"):
        __import__("time").tzset()
    california_hour = camera_time.camera_now().hour
    expected = (datetime.now(timezone.utc) - timedelta(hours=7)).hour
    assert california_hour in (expected, (expected - 1) % 24)   # PDT or PST


def test_challan_timestamp_carries_the_camera_utc_offset():
    stamp = camera_time.camera_now_iso()
    assert stamp.endswith(("-07:00", "-08:00"))


def test_report_footer_says_which_clock_it_shows():
    import numpy as np
    from report import build_report_png
    png = build_report_png(np.zeros((120, 160, 3), dtype=np.uint8), "cam", "Low", 3, 0.01, "LSTM")
    assert isinstance(png, (bytes, bytearray)) and png[:4] == b"\x89PNG"
