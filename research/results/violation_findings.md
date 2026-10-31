# Violation rules on recorded real traffic (2026-10-31)

Four clips of daytime freeway traffic, 4 minutes each (16 minutes in total), recorded with `research/record_clip.py`
from Caltrans cameras I-5 Broadway (Santa Ana), I-5 Crescent (Anaheim), I-210 Citrus (Azusa) and I-5 Olive (Burbank),
at 15, 30 and 50 fps. Every flagged image was looked at by eye.

| Version | Speeding | Wrong way | Stopped | Tailgating + lane change | What was wrong |
|---|---|---|---|---|---|
| First (as committed in 930c6a7) | 94 in 12 min* | 42 | 0 | 8 | Oncoming cars flagged as wrong-way (flow learned from a few early vehicles); speeds noisy; 50 fps clip sampled 3x as often |
| Flow learned from 15+ distinct vehicles, sampled by time | 63 | 0 | 2 | 65 | Speed noise; lateral noise |
| Final (speed-noise check, experimental rules off) | 0 | 2 | 3 | off | 5 flags, all looked like non-violations on review |

\* the first run covered three of the four clips.

**Speed noise** (`research/speed_diag.py`): per-vehicle speed estimates have percentiles 5/25/50/75/95 of
22/49/60/71/84 mph (Crescent) and 5/42/57/67/83 mph (Olive), i.e. a robust spread of about +-15 to 18 mph. With
"limit 65 + margin 15", 6-10% of ordinary vehicles exceeded 80 mph. A speeding flag now also requires typical speed
+ 2 robust standard deviations, which on these cameras is above the plausibility cap, so no speeding challans are issued
(the page says so; Demo mode skips the check).

**Wrong-way recall** (`research/inject_wrong_way.py`): a real vehicle image was driven backwards along a real lane path,
three times per clip, at least 70 s apart: **7 of 12 injections flagged**, 1 flag outside the injection windows.

**Conclusion.** With public low-resolution cameras, speeding, tailgating and lane-change enforcement are not reliable;
wrong-way detection works partly (7/12 on injected cases) but still raised false flags on plain footage, and the
stopped-vehicle rule cannot tell a freeway lane from a frontage road or lot in view. All of these are off by default in
the app and described as experimental. 16 minutes is a small sample; the review CSV is in `violation_review.csv`.
