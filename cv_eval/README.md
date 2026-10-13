# CV evaluation set

Hand-labelled live Caltrans camera frames used to measure the vision part of the app
(vehicle counting, density -> congestion label). See `evaluate_cv.py` for the metrics.

- `collect_frames.py` downloads a stratified sample (same number of random cameras from
  each district, plus extra Interstate cameras from busy districts) into `frames/`, indexed
  in `frames.csv`. All frames were captured on 2026-10-13 around 09:00-09:25 PDT, so the set
  is daytime only: no night, rain or fog.
- `labels.csv`: per frame, the vehicle `count`, the `congestion` class (Low / Moderate /
  High / Severe) and whether the frame is `usable` (with a `note` when it is not).
- `make_sheets.py` builds contact sheets (`--labels` overlays the labels) for review.
- `evaluate_cv.py` runs the sweep and writes `results/report.md`.

## How the labels were made, and their limits

The counts and classes were produced by looking at the frames (done with Claude's help),
not taken from a sensor or a published dataset. Treat them as approximate:

- Counts are +/- a few vehicles. Small, distant vehicles are counted when identifiable, which
  makes the labels generous and the detector look like it undercounts.
- Congestion classes are a visual judgement of how crowded the road looks, so they are
  subjective. There are only 2 Severe and 8 High frames of 124 usable ones, so per-class
  results for those are weak evidence.
- Review before relying on them: `python cv_eval/make_sheets.py --labels` overlays each
  label on its frame. Correct `labels.csv` and re-run the evaluation if you disagree.
- 33 of the 162 frames were Caltrans "Temporarily Unavailable" placeholders and 4 more were
  unusable (glare, haze, a parking lot); they are marked `usable=no` and excluded.
