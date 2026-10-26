"""Retrain and compare the historical models on California PeMS flow (A1), with the same baselines as baselines.py.

Uses PeMS 5-minute station files aggregated to hourly flow for the mainline stations nearest to the chosen
cameras (or for explicit --station ids). Features are calendar encodings plus lags of the flow itself
(PeMS has no weather; weather could be joined from Open-Meteo's archive in a later step). The split is
chronological 70/15/15, exactly as for the Minneapolis data, so the two sets of results are comparable.

    python research/pems_train.py --meta d07_meta.txt --stations d07_text_station_5min_2025_*.txt.gz \
        --station 716091 --station 716092
"""
import argparse
import glob
import os
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import pems  # noqa: E402

RESULTS = os.path.join(HERE, "results")
FEATURES = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend", "lag_1h", "lag_3h", "lag_24h", "rolling_mean_3h"]


def hourly_flow(stations: pd.DataFrame, ids) -> pd.Series:
    sub = stations[stations["station"].isin(ids)]
    per_5min = sub.groupby("ts")["flow"].sum(min_count=1)
    hourly = per_5min.resample("1h").sum(min_count=6)       # need at least half the bins observed
    return hourly.dropna()


def make_frame(hourly: pd.Series) -> pd.DataFrame:
    full = hourly.reindex(pd.date_range(hourly.index.min(), hourly.index.max(), freq="h")).interpolate(limit=3)
    df = pd.DataFrame({"y": full})
    df["hour"], df["dow"] = df.index.hour, df.index.dayofweek
    df["hour_sin"], df["hour_cos"] = np.sin(2 * np.pi * df.hour / 24), np.cos(2 * np.pi * df.hour / 24)
    df["dow_sin"], df["dow_cos"] = np.sin(2 * np.pi * df.dow / 7), np.cos(2 * np.pi * df.dow / 7)
    df["is_weekend"] = (df.dow >= 5).astype(int)
    df["lag_1h"], df["lag_3h"], df["lag_24h"] = full.shift(1), full.shift(3), full.shift(24)
    df["lag_168h"] = full.shift(168)
    df["rolling_mean_3h"] = full.shift(1).rolling(3).mean()
    return df.dropna()


def evaluate(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--meta", required=True)
    ap.add_argument("--stations", nargs="+", required=True)
    ap.add_argument("--station", type=int, action="append", required=True, help="station id(s) to sum (repeat the flag)")
    args = ap.parse_args()
    paths = sorted({p for pat in args.stations for p in glob.glob(pat)})
    st = pd.concat([pems.parse_station_5min(p) for p in paths], ignore_index=True)
    st = st[st["lane_type"].astype(str).str.upper() == "ML"]
    df = make_frame(hourly_flow(st, args.station))
    n = len(df)
    if n < 24 * 28:
        print(f"warning: only {n} usable hours; 8+ weeks are needed for a meaningful weekly baseline")
    n_test, n_val = int(n * 0.15), int(n * 0.15)
    train, test = df.iloc[: n - n_val - n_test], df.iloc[n - n_test:]
    y = test["y"].to_numpy()
    prof = train.groupby(["hour", "dow"])["y"].mean()
    preds = {
        "Persistence (last hour)": test["lag_1h"],
        "Seasonal naive (24 h)": test["lag_24h"],
        "Seasonal naive (1 week)": test["lag_168h"],
        "Hour x weekday mean": [prof.get((h, d), train["y"].mean()) for h, d in zip(test["hour"], test["dow"])],
    }
    sc = StandardScaler().fit(train[FEATURES])
    preds["LinearRegression"] = LinearRegression().fit(sc.transform(train[FEATURES]), train["y"]).predict(sc.transform(test[FEATURES]))
    preds["RandomForest"] = RandomForestRegressor(200, min_samples_leaf=3, n_jobs=-1, random_state=42).fit(train[FEATURES], train["y"]).predict(test[FEATURES])
    preds["GradientBoosting"] = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=42).fit(train[FEATURES], train["y"]).predict(test[FEATURES])
    rows = [{"Model": k, **evaluate(y, np.asarray(v))} for k, v in preds.items()]
    out = pd.DataFrame(rows)
    os.makedirs(RESULTS, exist_ok=True)
    out.to_csv(f"{RESULTS}/pems_models.csv", index=False)
    print(f"{n} hours ({df.index.min():%Y-%m-%d} to {df.index.max():%Y-%m-%d}); test hours {len(test)}")
    print(out.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
