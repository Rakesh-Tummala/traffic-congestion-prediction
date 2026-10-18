"""The clock the models should use: California time, where the cameras are.

The historical models take an hour of day, a weekday and a month. Using the computer's own clock
is wrong whenever it is in another time zone: at 03:46 in California a computer set to India time
(16:17) would ask the models about Sunday rush hour and report a false "unusual congestion"."""
from datetime import datetime
from zoneinfo import ZoneInfo

CAMERA_TZ = ZoneInfo("America/Los_Angeles")


def camera_now() -> datetime:
    """Current wall-clock time in California as a naive datetime (what the models expect)."""
    return datetime.now(CAMERA_TZ).replace(tzinfo=None)


def camera_now_iso() -> str:
    """Current California time with its UTC offset, for records that must be unambiguous."""
    return datetime.now(CAMERA_TZ).isoformat(timespec="seconds")
