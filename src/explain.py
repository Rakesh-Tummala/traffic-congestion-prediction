"""Best-effort explanation of a single congestion prediction: which input
features most influenced it.

Linear Regression is linearly decomposable, so its explanation is exact:
coefficient x scaled feature value, in the model's own scaled space. SVR
(RBF kernel) and the LSTM are not linearly decomposable, so for those we
instead report which features are most unusual right now compared to the
historical training distribution (a z-score against that feature's
training-set mean/std) — a proxy for "what's different about current
conditions", not a strict per-model attribution. It's honest about which
kind of explanation it's giving rather than presenting a rough proxy as if
it were exact.
"""
import os
from functools import lru_cache

import joblib
import numpy as np
import pandas as pd

from data_loader import FEATURE_COLUMNS, load_dataset

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

FEATURE_LABELS = {
    "hour_sin": "time of day", "hour_cos": "time of day",
    "dow_sin": "day of week", "dow_cos": "day of week",
    "month_sin": "month", "month_cos": "month",
    "is_weekend": "weekend", "temp_c": "temperature", "rain_1h": "rain",
    "snow_1h": "snow", "clouds_all": "cloud cover", "is_holiday": "holiday",
    "weather_code": "weather condition",
    "lag_1h": "traffic 1h ago", "lag_3h": "traffic 3h ago",
    "lag_24h": "traffic 24h ago", "rolling_mean_3h": "recent traffic trend",
}


@lru_cache(maxsize=1)
def _training_stats() -> pd.DataFrame:
    """Mean/std of each feature over the full training set — the 'typical'
    baseline a live reading gets compared against. Computed on a winsorized
    copy (clipped to the 0.5th-99.5th percentile) rather than the raw
    values: this dataset's rain_1h has a known data-quality outlier
    (one row logs 9831mm of rain in an hour — a sensor/logging error, not a
    real storm) that inflates the raw std enough to make an actual heavy
    downpour look barely unusual. Winsorizing keeps a handful of genuine
    outliers from making every other reading look artificially 'normal'.
    Zero-variance features (shouldn't occur here, but guarded) fall back to
    std=1 to avoid a divide-by-zero blowing up the z-score."""
    _, X, _ = load_dataset()
    lo = X.quantile(0.005)
    hi = X.quantile(0.995)
    clipped = X.clip(lower=lo, upper=hi, axis=1)
    stats = pd.DataFrame({"mean": clipped.mean(), "std": clipped.std()})
    stats["std"] = stats["std"].replace(0, 1.0)
    return stats


def explain_linear_regression(X_row: pd.DataFrame, top_n: int = 4) -> list[dict]:
    bundle = joblib.load(os.path.join(MODELS_DIR, "linear_regression.joblib"))
    X_scaled = bundle["scaler"].transform(X_row[bundle["features"]])[0]
    coefs = bundle["model"].coef_
    contributions = [
        {"feature": f, "label": FEATURE_LABELS.get(f, f), "contribution": float(c * v)}
        for f, c, v in zip(bundle["features"], coefs, X_scaled)
    ]
    contributions.sort(key=lambda d: -abs(d["contribution"]))
    return contributions[:top_n]


def explain_by_deviation(X_row: pd.DataFrame, top_n: int = 4) -> list[dict]:
    stats = _training_stats()
    deviations = []
    for f in FEATURE_COLUMNS:
        value = float(X_row[f].iloc[0])
        z = (value - stats.loc[f, "mean"]) / stats.loc[f, "std"]
        # Capped after computing magnitude-based rank order below, purely so a
        # freak value doesn't produce a meaningless "312 standard deviations"
        # in the UI — real ranking among ordinary values is unaffected since
        # the cap only ever engages for values that were already extreme.
        deviations.append({"feature": f, "label": FEATURE_LABELS.get(f, f), "z_score": float(np.clip(z, -8.0, 8.0))})
    deviations.sort(key=lambda d: -abs(d["z_score"]))
    return deviations[:top_n]


def explain_prediction(model_name: str, X_row: pd.DataFrame) -> dict:
    """Top contributing (Linear Regression) or most-unusual (SVR/LSTM)
    features for a prediction. `kind` tells the caller which flavor of
    explanation it got so the UI can phrase it honestly."""
    if model_name == "linear_regression":
        return {"kind": "exact", "top_features": explain_linear_regression(X_row)}
    return {"kind": "deviation", "top_features": explain_by_deviation(X_row)}
