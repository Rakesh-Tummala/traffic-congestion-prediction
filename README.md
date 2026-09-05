# Smart Traffic Congestion Prediction

Predicts road congestion by combining historical time/weather/recent-volume
patterns with live computer-vision vehicle counts from real public traffic
cameras, using three modeling techniques:

- **Linear Regression** and **Support Vector Regression (SVR)** — trained on
  hourly traffic volume, weather, and recent-history (lag) features
  (`src/train_regression.py`).
- **LSTM (deep learning)** — sequence model that learns from the last 24 hours
  of traffic to forecast the next hour, with validation-based early stopping
  (`src/train_dl.py`).
- **Computer Vision (YOLOv8s)** — detects and counts vehicles in a camera
  frame (upload, snapshot URL, webcam, or a real live Caltrans camera) and
  estimates road occupancy density (`src/cv_module.py`).

`src/live_predict.py` and `app.py` fuse the CV density signal with the trained
regression model to produce a live congestion estimate (Low / Moderate / High
/ Severe): whichever signal indicates worse congestion wins (see
[Fusion design](#fusion-design-cv--historical-model) below).

## Model quality

| Model               | MAE    | RMSE   | R²    |
|---------------------|-------:|-------:|------:|
| Linear Regression   | 334.2  | 451.9  | 0.948 |
| SVR (tuned rbf)     | 253.5  | 397.5  | 0.960 |
| LSTM (24h lookback) | ~200   | ~295   | ~0.977 |

(LSTM numbers vary slightly run-to-run from random weight init and early-stopping timing.)

(Regenerate with `cd src && python evaluate.py` — trains all three and writes
`models/model_comparison.png`.)

Two things drove the biggest accuracy gains over an initial time+weather-only
baseline (which capped LR/SVR at R²≈0.71–0.77):

1. **Cyclical time encoding** — `hour`/`day_of_week`/`month` as sin/cos pairs
   instead of raw integers, so hour 23 and hour 0 are correctly "close" to a
   linear model.
2. **Lag/rolling features** — `lag_1h`, `lag_3h`, `lag_24h`, `rolling_mean_3h`
   computed from the sensor's own recent history. This is the single biggest
   lever (LR: 0.71 → 0.95, SVR: 0.77 → 0.96) and is what let LR/SVR compete
   with the LSTM's implicit 24h lookback. Computed via a continuous-hourly
   reindex + interpolation so a lag next to one of the dataset's real gaps
   (~6.5% of hours) still estimates the true value 1h prior, not whatever
   the previous *row* happened to be (`data_loader.add_lag_features`).

SVR's hyperparameters (`C=100, epsilon=10, gamma='auto'`) were chosen via
`RandomizedSearchCV` with `TimeSeriesSplit` cross-validation over a 4000-row
subsample — see `notebooks/tune_svr.py`.

## Fusion design (CV ↔ historical model)

The regression/LSTM models are trained on historical sensor data; the CV
model detects vehicles independently and maps its density ratio to a
Low/Moderate/High/Severe label via fixed thresholds. No public dataset pairs
camera frames with this sensor's volume readings, so there's no ground truth
to *learn* a fusion weight from.

An earlier version nudged the model's historical baseline by a small
multiplicative factor based on the live density ratio. Testing against a real
gridlocked-highway photo exposed why that failed: the CV heuristic correctly
read "Severe" (density ratio 0.8), but the multiplicative nudge was too weak
to move the final label out of "Low" — the historical baseline simply
drowned out the live signal. The fusion now reports **whichever of the two
independent labels is more severe** — deliberately conservative, since
under-reporting congestion that's plainly visible on camera is a worse
failure mode than over-reporting it (`live_predict.fuse_labels`, covered by
`tests/test_fusion.py`).

When a real live volume sensor reading isn't available for the lag features
(the common case for a camera-only location), the model falls back to a
seasonal (hour, day-of-week) average fit on the training data
(`data_loader.seasonal_profile`, `live_predict.estimate_lag_features`). Pass
real readings via `--lag-1h/--lag-3h/--lag-24h` (CLI) or the "Advanced" panel
(dashboard) when a sensor is present — this is strictly more accurate.

## Live camera integration

`src/live_cameras.py` browses Caltrans's public CCTV network — real,
currently in-service cameras, no API key required
([cwwp2.dot.ca.gov](https://cwwp2.dot.ca.gov/documentation/cctv/cctv.htm): "There is no
charge for the use of this data"). Covers all 12 Caltrans districts; district
7 (LA) alone has ~480 in-service cameras. The dashboard's "Live Caltrans
camera" tab lists and fetches from these directly.

```bash
cd src
python live_cameras.py 7 --search "I-5" --limit 10   # list LA-area I-5 cameras
```

**Known limitation, verified against a real live nighttime frame**: a
generic COCO-pretrained YOLO struggles on these cameras' low-resolution
(~320×260), often-nighttime images — 0 vehicles detected on a real frame with
visible cars. Tried and rejected three fixes during development: lowering the
confidence threshold (produced false positives in the wrong location, not
better recall), upscaling the frame, and CLAHE contrast enhancement (amplified
sensor noise instead of revealing detail). The fusion logic tolerates this
gracefully — when CV under-detects, the historical model's seasonal/lag-based
estimate still drives the final label — but real accuracy improvement here
would need either higher-resolution daytime-oriented cameras or a detector
fine-tuned on low-light traffic imagery (future work).

The CV density heuristic also reads occupancy purely from bounding-box area,
so an extreme close-up of a single vehicle can misread as "Severe" the same
as a genuinely packed road — not an issue for its intended input (a
roadside/overhead camera with a fixed wide field of view), but worth noting.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only build
pip install -r requirements.txt
```

## Train

```bash
cd src
python train_regression.py      # Linear Regression + SVR -> models/
python train_dl.py              # LSTM -> models/
python evaluate.py              # trains everything, prints + plots comparison
```

## Test

```bash
python -m pytest tests/ -v
```

Covers the fusion logic (the "Low" bug and its fix), the data/feature
pipeline (lag correlation, cyclical encoding, no missing values), and the
live-inference path (all three models return sane predictions, a real
sensor reading actually changes the output vs. the seasonal fallback).

## Run computer vision on a single frame

```bash
cd src
python cv_module.py path/to/image.jpg --save annotated.jpg
python cv_module.py https://example-dot-traffic-cam.gov/snapshot.jpg
python cv_module.py 0            # webcam
```

## Fused live prediction (CLI)

```bash
cd src
python live_predict.py path/to/image.jpg --weather-main Clear --temp-c 22 --model svr
# with a real sensor reading, if this location has one:
python live_predict.py path/to/image.jpg --lag-1h 5200 --lag-3h 4900 --lag-24h 5000
```

## Dashboard

```bash
streamlit run app.py
```

Pick a real live Caltrans camera (or upload an image / paste a snapshot URL),
set weather conditions, and get:

- A color-coded **congestion badge** (green/amber/orange/red) for the final
  fused level.
- **All three models side by side** (Linear Regression, SVR, LSTM) with
  predicted volume and R², not just whichever one is "selected" — the LSTM
  is the most accurate (R²=0.977) but needs a synthetic 24h feature sequence
  built the same way the live lag-feature fallback works
  (`live_predict.predict_lstm_volume`), so it wasn't wired into live inference
  until this pass; previously only LR/SVR were reachable from the app despite
  the LSTM being the best model.
- A **24-hour forecast chart** (Altair) showing predicted volume across the
  whole day with the four congestion bands shaded in the background and the
  current hour marked — context for "is right now unusual," not just a bare
  number (`live_predict.forecast_day`).
- A **live map** (pydeck) of the selected district's cameras, with the
  chosen one highlighted, so you can see where you're looking before fetching.

Model, CV, and LSTM weights are all cached across reruns
(`lru_cache`/`st.cache_data`) so the dashboard doesn't reload them from disk
on every interaction, and camera/forecast fetches are cache-backed with a
TTL so switching sliders back and forth doesn't refire live network calls.

## Project structure

```
data/                UCI traffic dataset
models/               trained model artifacts (.joblib, .pt) + comparison chart
notebooks/
  tune_svr.py         one-off SVR hyperparameter search (RandomizedSearchCV)
src/
  data_loader.py      dataset loading + feature engineering (cyclical time, weather, lag features)
  train_regression.py Linear Regression + SVR (+ seasonal_profile for live fallback)
  train_dl.py         LSTM (PyTorch), validation-based early stopping
  cv_module.py        YOLOv8s vehicle detection/counting
  live_cameras.py     browse/fetch real public Caltrans CCTV camera snapshots
  live_predict.py     fuses CV output + trained models -> congestion label
  evaluate.py         trains all 3 models, prints/plots MAE/RMSE/R2 comparison
tests/                pytest suite (fusion logic, data pipeline, live inference)
app.py                Streamlit dashboard
```
