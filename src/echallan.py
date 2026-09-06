"""Show what a traffic fine WOULD HAVE BEEN for a vehicle flagged as speeding
by speed_estimation.check_speeds() — purely informational, never an actual
charge.

This is an educational/demo feature only: it reports what a citation styled
after India's e-challan system would show (the fine schedule below
references India's Motor Vehicles (Amendment) Act, 2019, Section 183, which
publicly sets penalties for over-speeding) purely to illustrate how a
detection pipeline could feed into a citation workflow. No real citation is
issued and no fine is charged — it is not connected to any government
system, vehicle registry, or payment processor. The vehicle number can come
from src/anpr.py's automatic plate-recognition attempt (clearly marked as
such, with its confidence) or be typed in manually; either way it's never
harvested or stored beyond this local, gitignored log. Every generated
document carries a prominent disclaimer banner restating all of this.
"""
import json
import os
import uuid
from dataclasses import MISSING, asdict, dataclass, fields
from datetime import datetime
from html import escape

DISCLAIMER = (
    "SIMULATED — SHOWS WHAT THE FINE WOULD HAVE BEEN, FOR DEMONSTRATION / EDUCATIONAL "
    "PURPOSES ONLY. No real citation was issued and no fine was charged. Not connected to "
    "any government authority, vehicle registry, or payment system."
)

# Illustrative only — India's Motor Vehicles (Amendment) Act, 2019, Section 183
# sets over-speeding penalties; actual amounts vary by state and circumstances.
# "aggravated" here is an arbitrary demo threshold (20 mph over the limit),
# not a legal definition.
FINE_SCHEDULE_INR = {
    "motorcycle": {"base": 1000, "aggravated": 2000},
    "car": {"base": 1000, "aggravated": 2000},
    "bus": {"base": 2000, "aggravated": 4000},
    "truck": {"base": 2000, "aggravated": 4000},
}
AGGRAVATED_OVER_MPH = 20.0
LOG_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "echallan_log.json")


@dataclass
class Challan:
    challan_id: str
    issued_at: str
    camera_name: str
    vehicle_class: str
    vehicle_number: str
    speed_mph: float
    speed_limit_mph: float
    over_mph: float
    fine_inr: int
    legal_reference: str
    plate_auto_detected: bool = False
    plate_confidence: float = 0.0


def compute_fine(vehicle_class: str, over_mph: float) -> tuple[int, str]:
    """Illustrative fine amount + the (real, public) law section it
    references. Unknown vehicle classes fall back to the car/LMV slab."""
    slab = FINE_SCHEDULE_INR.get(vehicle_class, FINE_SCHEDULE_INR["car"])
    amount = slab["aggravated"] if over_mph >= AGGRAVATED_OVER_MPH else slab["base"]
    return amount, "Motor Vehicles (Amendment) Act, 2019 — Section 183 (illustrative)"


def generate_challan(speed_result: dict, camera_name: str, speed_limit_mph: float,
                      vehicle_number: str = "", plate_auto_detected: bool = False,
                      plate_confidence: float = 0.0) -> Challan:
    """Build a Challan from one entry of speed_estimation.check_speeds()'s
    `results` list. Raises ValueError if the vehicle wasn't actually flagged
    as speeding — a challan should never be generated for a non-violation."""
    if not speed_result.get("speeding"):
        raise ValueError("Refusing to generate a challan for a vehicle that wasn't flagged as speeding.")

    fine_inr, legal_ref = compute_fine(speed_result["cls_name"], speed_result["over_mph"])
    return Challan(
        challan_id=f"DEMO-{uuid.uuid4().hex[:10].upper()}",
        issued_at=datetime.now().isoformat(timespec="seconds"),
        camera_name=camera_name,
        vehicle_class=speed_result["cls_name"],
        vehicle_number=vehicle_number.strip() or "NOT ENTERED (demo)",
        speed_mph=round(speed_result["speed_mph"], 1),
        speed_limit_mph=speed_limit_mph,
        over_mph=round(speed_result["over_mph"], 1),
        fine_inr=fine_inr,
        legal_reference=legal_ref,
        plate_auto_detected=plate_auto_detected,
        plate_confidence=round(plate_confidence, 3),
    )


def render_challan_html(challan: Challan) -> str:
    c = challan
    if c.plate_auto_detected:
        plate_note = f"auto-detected, confidence {c.plate_confidence:.2f} — unverified"
    else:
        plate_note = "entered manually"
    return f"""
<div style="border:2px solid #ef4444;border-radius:10px;padding:16px;font-family:sans-serif;
max-width:480px;background:#1a0f0f;">
  <div style="background:#ef4444;color:white;font-weight:700;text-align:center;
  padding:6px;border-radius:6px;margin-bottom:12px;font-size:12px;letter-spacing:0.5px;">
    {escape(DISCLAIMER)}
  </div>
  <h3 style="margin:0 0 8px 0;color:#f3f4f6;">Speed Violation Summary (Simulated)</h3>
  <table style="width:100%;color:#d1d5db;font-size:14px;border-collapse:collapse;">
    <tr><td style="padding:3px 0;opacity:0.7;">Reference No.</td><td><b>{escape(c.challan_id)}</b></td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Recorded</td><td>{escape(c.issued_at)}</td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Location</td><td>{escape(c.camera_name)}</td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Vehicle</td><td>{escape(c.vehicle_class.title())} — {escape(c.vehicle_number)}
    <div style="font-size:11px;opacity:0.6;">({escape(plate_note)})</div></td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Violation</td><td>Over-speeding</td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Detected speed</td><td>{c.speed_mph:.1f} mph</td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Speed limit</td><td>{c.speed_limit_mph:.0f} mph</td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Over limit by</td><td>{c.over_mph:.1f} mph</td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Fine that would have applied</td><td><b>₹{c.fine_inr}</b> (not charged)</td></tr>
    <tr><td style="padding:3px 0;opacity:0.7;">Reference law</td><td style="font-size:12px;">{escape(c.legal_reference)}</td></tr>
  </table>
</div>
"""


def _load_log() -> list[dict]:
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8") as f:
        entries = json.load(f)
    # Backfill fields added after some entries were already logged (e.g.
    # plate_auto_detected/plate_confidence) so older log files don't crash
    # code that expects every Challan field to be present.
    defaults = {f.name: f.default for f in fields(Challan) if f.default is not MISSING}
    for entry in entries:
        for key, default in defaults.items():
            entry.setdefault(key, default)
    return entries


def append_to_log(challan: Challan) -> None:
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log = _load_log()
    log.append(asdict(challan))
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)


def load_log() -> list[dict]:
    return _load_log()
