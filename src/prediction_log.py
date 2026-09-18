"""Local log of live predictions, kept purely so the dashboard can show a
recent trend for a camera instead of only "right now" — not a real
telemetry/analytics pipeline, just a gitignored JSON file for this install."""
import json
import os
from datetime import datetime

LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "prediction_log.json")
MAX_LOG_ENTRIES = 5000  # oldest entries are dropped once the log grows past this


def _load_log() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def log_prediction(camera_name: str, model_name: str, volume: float, model_label: str,
                    cv_label: str, final_label: str) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log = _load_log()
    log.append({
        "camera_name": camera_name,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "model": model_name,
        "volume": round(volume, 1),
        "model_label": model_label,
        "cv_label": cv_label,
        "final_label": final_label,
    })
    if len(log) > MAX_LOG_ENTRIES:
        log = log[-MAX_LOG_ENTRIES:]
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def load_history(camera_name: str = None, limit: int = 200) -> list[dict]:
    """Most recent logged predictions, optionally filtered to one camera,
    oldest first (ready to feed straight into a trend chart)."""
    log = _load_log()
    if camera_name:
        log = [e for e in log if e["camera_name"] == camera_name]
    return log[-limit:]
