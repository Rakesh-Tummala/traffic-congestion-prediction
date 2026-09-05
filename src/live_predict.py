"""Fusion pipeline: take a live camera frame + current time/weather, run CV vehicle
counting, then feed the combined feature vector into the trained models (Linear
Regression, SVR, and LSTM) to produce a live congestion prediction."""
import argparse
import os
from datetime import datetime, timedelta
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd
import torch

from cv_module import CONGESTION_LEVELS, analyze_frame, congestion_from_density, load_frame
from data_loader import FEATURE_COLUMNS, WEATHER_CODES, load_dataset
from train_dl import SEQ_LEN, TrafficLSTM

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
REGRESSION_MODELS = ["linear_regression", "svr"]
ALL_MODELS = REGRESSION_MODELS + ["lstm"]


@lru_cache(maxsize=1)
def _seasonal_profile():
    return joblib.load(os.path.join(MODELS_DIR, "seasonal_profile.joblib"))


def _seasonal_estimate(profile: pd.Series, at: datetime) -> float:
    """Typical volume at this hour/day-of-week, for when no real-time sensor
    reading is available. Falls back to the global mean if that exact
    (hour, day_of_week) slot wasn't in the training data."""
    try:
        return float(profile.loc[(at.hour, at.weekday())])
    except KeyError:
        return float(profile.mean())


def estimate_lag_features(now: datetime, lag_1h: float = None, lag_3h: float = None,
                           lag_24h: float = None, rolling_mean_3h: float = None) -> dict:
    """Recent-history features for the model. Pass real sensor readings when a live
    volume feed is available (most accurate); any left as None fall back to a
    seasonal (hour, day-of-week) average — a reasonable estimate for a camera-only
    deployment with no volume sensor, but strictly worse than a real reading."""
    profile = _seasonal_profile()
    return {
        "lag_1h": lag_1h if lag_1h is not None else _seasonal_estimate(profile, now - timedelta(hours=1)),
        "lag_3h": lag_3h if lag_3h is not None else _seasonal_estimate(profile, now - timedelta(hours=3)),
        "lag_24h": lag_24h if lag_24h is not None else _seasonal_estimate(profile, now - timedelta(hours=24)),
        "rolling_mean_3h": rolling_mean_3h if rolling_mean_3h is not None else np.mean([
            _seasonal_estimate(profile, now - timedelta(hours=h)) for h in (1, 2, 3)
        ]),
    }


def build_feature_row(now: datetime, temp_c: float, rain_1h: float, snow_1h: float,
                       clouds_all: float, is_holiday: int, weather_main: str,
                       lag_1h: float = None, lag_3h: float = None,
                       lag_24h: float = None, rolling_mean_3h: float = None) -> pd.DataFrame:
    hour, dow, month = now.hour, now.weekday(), now.month
    row = {
        "hour_sin": np.sin(2 * np.pi * hour / 24), "hour_cos": np.cos(2 * np.pi * hour / 24),
        "dow_sin": np.sin(2 * np.pi * dow / 7), "dow_cos": np.cos(2 * np.pi * dow / 7),
        "month_sin": np.sin(2 * np.pi * month / 12), "month_cos": np.cos(2 * np.pi * month / 12),
        "is_weekend": int(dow >= 5),
        "temp_c": temp_c, "rain_1h": rain_1h, "snow_1h": snow_1h, "clouds_all": clouds_all,
        "is_holiday": is_holiday, "weather_code": WEATHER_CODES.get(weather_main, -1),
        **estimate_lag_features(now, lag_1h, lag_3h, lag_24h, rolling_mean_3h),
    }
    return pd.DataFrame([row])[FEATURE_COLUMNS]


@lru_cache(maxsize=4)
def _load_model_bundle(model_path: str):
    return joblib.load(model_path)


def predict_volume(model_path: str, X: pd.DataFrame) -> float:
    bundle = _load_model_bundle(model_path)
    X_scaled = bundle["scaler"].transform(X[bundle["features"]])
    return float(bundle["model"].predict(X_scaled)[0])


@lru_cache(maxsize=1)
def _load_lstm():
    scalers = joblib.load(os.path.join(MODELS_DIR, "lstm_scalers.joblib"))
    model = TrafficLSTM(n_features=len(scalers["features"]))
    model.load_state_dict(torch.load(os.path.join(MODELS_DIR, "lstm.pt"), map_location="cpu"))
    model.eval()
    return model, scalers


def predict_lstm_volume(now: datetime, temp_c: float, rain_1h: float, snow_1h: float,
                         clouds_all: float, is_holiday: int, weather_main: str,
                         lag_1h: float = None, lag_3h: float = None, lag_24h: float = None) -> float:
    """Predict next-hour volume with the LSTM, which needs a 24-hour sequence of
    feature rows rather than a single snapshot. Without a live sensor feed there's
    no real per-hour history to supply, so each of the 24 past hours is built the
    same way build_feature_row's fallback works: current weather held constant
    across the window (a simplifying assumption — weather 20h ago likely differed,
    but the model has no other source for it) and lag features from the seasonal
    (hour, day-of-week) profile, seeded with any real recent readings supplied."""
    model, scalers = _load_lstm()
    rows = []
    for hours_ago in range(SEQ_LEN, 0, -1):
        at = now - timedelta(hours=hours_ago)
        row = build_feature_row(at, temp_c, rain_1h, snow_1h, clouds_all, is_holiday, weather_main,
                                 lag_1h=lag_1h if hours_ago == 1 else None,
                                 lag_3h=lag_3h if hours_ago == 1 else None,
                                 lag_24h=lag_24h if hours_ago == 1 else None)
        rows.append(row.iloc[0])

    X = pd.DataFrame(rows)[scalers["features"]]
    X_scaled = scalers["x_scaler"].transform(X).astype(np.float32)
    with torch.no_grad():
        pred_scaled = model(torch.from_numpy(X_scaled).unsqueeze(0)).numpy()
    return float(scalers["y_scaler"].inverse_transform(pred_scaled.reshape(-1, 1))[0, 0])


def predict_all_models(now: datetime, temp_c: float, rain_1h: float, snow_1h: float,
                        clouds_all: float, is_holiday: int, weather_main: str,
                        lag_1h: float = None, lag_3h: float = None, lag_24h: float = None) -> dict:
    """Run all three trained models and return {model_name: predicted_volume}."""
    X = build_feature_row(now, temp_c, rain_1h, snow_1h, clouds_all, is_holiday, weather_main,
                           lag_1h=lag_1h, lag_3h=lag_3h, lag_24h=lag_24h)
    predictions = {
        name: predict_volume(os.path.join(MODELS_DIR, f"{name}.joblib"), X)
        for name in REGRESSION_MODELS
    }
    predictions["lstm"] = predict_lstm_volume(now, temp_c, rain_1h, snow_1h, clouds_all,
                                               is_holiday, weather_main, lag_1h, lag_3h, lag_24h)
    return predictions


@lru_cache(maxsize=1)
def congestion_bins() -> np.ndarray:
    """Quartile edges of traffic_volume fit on the full training dataset, used to
    map a raw predicted volume to a Low/Moderate/High/Severe label."""
    _, _, y = load_dataset()
    bins = y.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy().copy()
    bins[0] -= 1
    bins[-1] += 1
    return bins


def calibrated_label(volume: float) -> str:
    bins = congestion_bins()
    for i in range(4):
        if bins[i] < volume <= bins[i + 1]:
            return CONGESTION_LEVELS[i]
    return "Severe" if volume > bins[-1] else "Low"


def fuse_labels(model_label: str, cv_label: str) -> str:
    """Combine the historical time/weather model's label with the live CV density
    label by taking whichever is more severe.

    A weighted/multiplicative blend was tried first and failed badly in practice:
    on a real gridlocked-highway test photo the CV density heuristic correctly
    read "Severe" (density ratio 0.75+), but a small multiplicative nudge to the
    model's historical baseline was too weak to move the final label out of "Low".
    Taking the max-severity of the two independent signals is simple, transparent,
    and appropriate for a warning system: never report a lower congestion level
    than what the live camera plainly shows, even if history says the road is
    normally quiet at this hour.
    """
    return max([model_label, cv_label], key=CONGESTION_LEVELS.index)


def forecast_day(base_date: datetime, temp_c: float, rain_1h: float, snow_1h: float,
                  clouds_all: float, is_holiday: int, weather_main: str, model: str = "lstm") -> pd.DataFrame:
    """Predicted volume for every hour of base_date's calendar day, holding weather
    constant — gives context for "how does the current prediction compare to the
    rest of the day" rather than just a single number in isolation."""
    rows = []
    for hour in range(24):
        at = base_date.replace(hour=hour, minute=0, second=0, microsecond=0)
        if model == "lstm":
            volume = predict_lstm_volume(at, temp_c, rain_1h, snow_1h, clouds_all, is_holiday, weather_main)
        else:
            X = build_feature_row(at, temp_c, rain_1h, snow_1h, clouds_all, is_holiday, weather_main)
            volume = predict_volume(os.path.join(MODELS_DIR, f"{model}.joblib"), X)
        rows.append({"hour": hour, "volume": volume, "label": calibrated_label(volume)})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser(description="Live congestion prediction fusing a camera frame with trained models.")
    parser.add_argument("source", help="Image path, video path, snapshot URL, or webcam index")
    parser.add_argument("--temp-c", type=float, default=20.0)
    parser.add_argument("--rain-1h", type=float, default=0.0)
    parser.add_argument("--snow-1h", type=float, default=0.0)
    parser.add_argument("--clouds-all", type=float, default=20.0)
    parser.add_argument("--is-holiday", type=int, default=0)
    parser.add_argument("--weather-main", default="Clear", choices=list(WEATHER_CODES.keys()))
    parser.add_argument("--model", default="lstm", choices=ALL_MODELS,
                        help="Which historical model drives the fused label (default: lstm, the most accurate)")
    parser.add_argument("--save-annotated", default=None)
    parser.add_argument("--lag-1h", type=float, default=None, help="Real sensor volume reading from 1h ago, if available")
    parser.add_argument("--lag-3h", type=float, default=None, help="Real sensor volume reading from 3h ago, if available")
    parser.add_argument("--lag-24h", type=float, default=None, help="Real sensor volume reading from 24h ago, if available")
    args = parser.parse_args()

    frame = load_frame(args.source)
    cv_result = analyze_frame(frame)
    cv_label = congestion_from_density(cv_result["density_ratio"])
    print(f"[CV] vehicle_count={cv_result['vehicle_count']}  density_ratio={cv_result['density_ratio']}  "
          f"image-based congestion={cv_label}")

    now = datetime.now()
    predictions = predict_all_models(now, args.temp_c, args.rain_1h, args.snow_1h, args.clouds_all,
                                      args.is_holiday, args.weather_main,
                                      lag_1h=args.lag_1h, lag_3h=args.lag_3h, lag_24h=args.lag_24h)
    print()
    for name, volume in predictions.items():
        print(f"[{name:>17s}] predicted volume: {volume:8.0f} -> {calibrated_label(volume)}")

    model_label = calibrated_label(predictions[args.model])
    final_label = fuse_labels(model_label, cv_label)
    print(f"\n[fused] Using {args.model} + live camera, more severe of the two signals: {final_label}")

    if args.save_annotated:
        import cv2
        cv2.imwrite(args.save_annotated, cv_result["annotated_frame"])
        print(f"Saved annotated frame to {args.save_annotated}")


if __name__ == "__main__":
    main()
