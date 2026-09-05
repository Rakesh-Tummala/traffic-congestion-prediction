# Smart Traffic Congestion Prediction

Predicts road congestion by combining historical time/weather/recent-volume
patterns with live computer-vision vehicle counts from real public traffic
cameras. Three modeling techniques (Linear Regression, SVR, LSTM) are trained
on historical sensor data; a YOLOv8s detector reads live camera frames; the
two are fused into a single Low/Moderate/High/Severe congestion estimate,
all wrapped in a Streamlit dashboard.

- **Live app**: `streamlit run app.py` → http://localhost:8501
- **Repo**: [github.com/Rakesh-Tummala/traffic-congestion-prediction](https://github.com/Rakesh-Tummala/traffic-congestion-prediction)

---

## How to use the app

1. **Install & launch** (see [Setup](#setup) below), then run:
   ```bash
   streamlit run app.py
   ```
2. **Give it a camera frame** — pick one of three tabs under "1. Camera frame":
   - **Live Caltrans camera** (recommended): choose a district, optionally
     filter by route/location, pick a camera from the dropdown — the map
     shows where it is (selected one highlighted in red) — then click
     **Fetch live snapshot** to pull the current real image.
   - **Upload image**: any traffic photo (JPG/PNG).
   - **Snapshot URL**: paste a direct image URL (any public traffic cam).
3. **Set conditions** — temperature, rain/snow, cloud cover, weather type,
   holiday flag. These feed the historical model; leave them at sensible
   defaults if you just want a quick look.
4. **Pick which model drives the forecast** — LSTM (most accurate, R²=0.977),
   SVR, or Linear Regression. All three run regardless; this just picks which
   one's label feeds the final fused result and the forecast chart.
5. *(Optional)* **Advanced panel** — if the camera's location has a real
   traffic volume sensor, enter its 1h/3h/24h-ago readings for a more accurate
   prediction than the default seasonal-average fallback.
6. **Read the result**:
   - A color-coded **congestion badge** (green→red) — the final answer.
   - The **detected-vehicles image** with bounding boxes drawn.
   - **Vehicle count** and **image density ratio** from the live frame.
   - **All three models** compared side by side (predicted volume, R²,
     level) — not just the one you picked.
   - **Today's forecast** — predicted volume for every hour of the day, with
     the four congestion bands shaded and the current hour marked, so you can
     see whether right now is unusual for this time of day.

Command-line equivalents (no dashboard) are documented further down under
[Run computer vision on a single frame](#run-computer-vision-on-a-single-frame)
and [Fused live prediction (CLI)](#fused-live-prediction-cli).

---

## Tech stack

| Layer                  | Technology |
|------------------------|------------|
| Language               | Python 3.11 |
| Classical ML           | scikit-learn — `LinearRegression`, `SVR` (rbf kernel, tuned via `RandomizedSearchCV` + `TimeSeriesSplit`), `StandardScaler` |
| Deep learning          | PyTorch — custom `TrafficLSTM` (2-layer LSTM + linear head), CPU inference |
| Computer vision        | Ultralytics **YOLOv8s** (pretrained on COCO) for vehicle detection, OpenCV for frame I/O |
| Data handling          | pandas, NumPy |
| Live camera data       | Caltrans public CCTV JSON feed (`cwwp2.dot.ca.gov`) via `requests` — no API key |
| Dashboard              | Streamlit |
| Charts / visualization | Altair (24h forecast chart), pydeck (camera location map), Matplotlib/Seaborn (offline model-comparison chart) |
| Testing                | pytest (12 tests: fusion logic, feature pipeline, live inference) |
| Model persistence      | joblib (sklearn models + scalers), native PyTorch `state_dict` (LSTM) |
| Training dataset       | [UCI Metro Interstate Traffic Volume](https://archive.ics.uci.edu/dataset/492/metro+interstate+traffic+volume) (~40k hourly readings, 2012–2018) |
| Version control        | Git, hosted on GitHub |

---

## System design

```mermaid
flowchart TD
    subgraph Sources["Data sources"]
        A1["UCI Traffic Volume dataset<br/>(historical hourly sensor + weather)"]
        A2["Live Caltrans camera<br/>(real-time JPEG snapshot)"]
    end

    subgraph Offline["Offline training (src/train_regression.py, train_dl.py)"]
        A1 --> B1["Feature engineering<br/>cyclical time + lag/rolling features<br/>(data_loader.py)"]
        B1 --> C1["Linear Regression"]
        B1 --> C2["SVR (tuned)"]
        B1 --> C3["LSTM (24h sequence)"]
        C1 & C2 & C3 --> D1["models/ artifacts<br/>.joblib / .pt + scalers + seasonal profile"]
    end

    subgraph Live["Live inference (src/live_predict.py)"]
        A2 --> E1["YOLOv8s vehicle detection<br/>(cv_module.py)"]
        E1 --> E2["Vehicle count + density ratio"]
        E2 --> F1["Density → congestion heuristic"]
        D1 --> F2["Historical model prediction<br/>(seasonal fallback for lag features<br/>if no real sensor reading given)"]
        F1 --> G["Fusion: report the MORE SEVERE<br/>of the two signals<br/>(fuse_labels)"]
        F2 --> G
    end

    G --> H["Streamlit dashboard (app.py)<br/>badge · model comparison · 24h forecast · camera map"]
```

**Why fusion takes the max, not a blend**: the historical model and the live
camera are independent signals with no shared ground truth to learn a
combining weight from. An earlier multiplicative blend let a real
gridlocked-highway test photo get reported as "Low" congestion because the
historical baseline drowned out the live signal. Taking whichever label is
more severe is simple, transparent, and appropriate for a warning system —
see [Fusion design](#fusion-design-cv--historical-model) for the full story.

**Why the LSTM needs special handling for live inference**: it was trained on
24-hour sequences, so a single live snapshot in time isn't enough input on
its own. `live_predict.predict_lstm_volume` builds a synthetic 24-hour
feature sequence the same way the lag-feature fallback works (seasonal
average per hour, or real sensor readings if supplied) and feeds that
through the trained model.

---

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

No equivalent official, no-signup Indian (or other-country) public camera
feed was found during development — third-party "aggregator" sites that
claim thousands of live cameras typically index feeds without clear
authorization from camera owners, unlike Caltrans which is the state DOT
publishing its own data with an explicit public-use policy, so none of those
were integrated. The **Upload image** / **Snapshot URL** tabs work with any
traffic photo or image URL regardless of country.

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

---

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
python live_predict.py path/to/image.jpg --weather-main Clear --temp-c 22 --model lstm
# with a real sensor reading, if this location has one:
python live_predict.py path/to/image.jpg --lag-1h 5200 --lag-3h 4900 --lag-24h 5000
```

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
