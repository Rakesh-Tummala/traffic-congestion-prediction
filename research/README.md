# Research evaluation

Everything needed to turn the app into a measured study. Run all commands from the project root.

| Question | Script | Needs | Output |
|---|---|---|---|
| Is 0.977 R² real, or just the lag features? Which models are actually better? | `baselines.py` | this repo | `results/results.md`, `main_comparison.csv`, `ablation.csv`, `horizon.csv`, `tscv.csv`, `bootstrap.csv`, `seed_spread.csv` |
| How wide should the forecast band be? | `conformal_intervals.py` | trained models | `models/conformal_intervals.json` (used by the app), coverage printed |
| All of the above in one go | `run_all.py` | this repo | as above |
| Does the camera agree with real loop detectors? (A3) | `log_cameras.py` then `pems_validation.py` | PeMS files | `results/pems_validation.md` |
| Is the speed estimate right? (A4) | `log_cameras.py --burst-every 3` then `pems_validation.py` | PeMS files | `pems_a4_speed.csv` |
| Does fusing history with the camera help? (A6) | `pems_validation.py` | PeMS files | `pems_a6_fusion.csv` |
| Do the models do better trained on California data? (A1) | `pems_train.py` | PeMS files | `pems_models.csv` |
| How good is the detector across the day? (A5) | `sample_for_labels.py`, label, `detector_eval.py` | logged frames + labels | `detector_by_time.csv` |
| Are the hand counts reliable? | `label_agreement.py` | a second annotator's counts | printed |
| How often do the violation rules fire, and how often are they right? (B1) | `record_clip.py`, `violation_eval.py replay`, review, `violation_eval.py score` | recorded clips + human review | `violation_rates.csv`, `violation_precision.csv` |
| Do the wrong-way rule's recall hold up? (B1) | `violation_eval.py reversal` | one recorded clip | `violation_reversal.csv` |

## PeMS (ground truth)

1. Register for a free account at <https://pems.dot.ca.gov>.
2. In the Data Clearinghouse choose **Station 5-Minute**, District 7, and download the days that overlap the camera log
   (also the station *metadata* file for that district). Files look like `d07_text_station_5min_2025_03_03.txt.gz`.
3. `python research/pems_validation.py --camera-log research/data/camera_log.csv --meta <meta file> --stations <station files>`.
4. `python research/pems_train.py --meta <meta file> --stations <station files> --station <id> [--station <id> ...]` (at least 8 weeks).

The readers in `pems.py` follow PeMS's published column layout and are tested on synthetic files in that layout.
Open the first real file, compare a few rows by eye with the PeMS website, and check `match_stations` picks sensible stations
for each camera before trusting any number.

## Logging cameras (needs the computer to stay on)

```
python research/log_cameras.py --district 7 --max-cameras 8 --hours 72 --burst-every 3 --save-frames
```
Writes a row per camera every 5 minutes (aligned to PeMS's bins), saves the JPEGs, and every third round watches 55 s of
video to record the raw ground-plane median speed. Keep it running over weekdays and a weekend, in day and night.

## Honest limits to carry into any paper

* The historical models are trained on Minneapolis data; only PeMS retraining (A1) makes a California claim.
* Detector labels were made by looking at frames (with Claude's help); a second human annotator and `label_agreement.py` are needed.
* Violation rules are validated by synthetic tests, a reversal-injection recall test, and a precision estimate from human review of
  flagged events. Natural wrong-way events are too rare to measure recall on. Tailgating and lane-change rules are experimental.
* Turn signals, red-light running and painted-line violations cannot be detected from these cameras and are not attempted.
