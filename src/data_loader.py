"""Load the Metro Interstate Traffic Volume dataset and engineer model features."""
import os

import numpy as np
import pandas as pd

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "Metro_Interstate_Traffic_Volume.csv")

FEATURE_COLUMNS = [
    "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
    "is_weekend", "temp_c", "rain_1h", "snow_1h", "clouds_all",
    "is_holiday", "weather_code",
    "lag_1h", "lag_3h", "lag_24h", "rolling_mean_3h",
]
LAG_COLUMNS = ["lag_1h", "lag_3h", "lag_24h", "rolling_mean_3h"]
TARGET_COLUMN = "traffic_volume"

WEATHER_CODES = {
    "Clear": 0, "Clouds": 1, "Mist": 2, "Rain": 3, "Drizzle": 4,
    "Snow": 5, "Thunderstorm": 6, "Fog": 7, "Haze": 8, "Smoke": 9, "Squall": 10,
}


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date_time"] = pd.to_datetime(df["date_time"])
    df = df.drop_duplicates(subset="date_time").sort_values("date_time").reset_index(drop=True)
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["hour"] = df["date_time"].dt.hour
    df["day_of_week"] = df["date_time"].dt.dayofweek
    df["month"] = df["date_time"].dt.month
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["is_holiday"] = (df["holiday"] != "None").astype(int)
    df["temp_c"] = df["temp"] - 273.15  # dataset stores Kelvin
    df["weather_code"] = df["weather_main"].map(WEATHER_CODES).fillna(-1).astype(int)
    return df


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add recent-history features (lag_1h, lag_3h, lag_24h, rolling_mean_3h) computed
    from the target itself. ~6.5% of hourly gaps in this sensor's readings are bridged
    by reindexing to a continuous hourly series and linearly interpolating before
    computing lags, so a lag_1h next to a data gap still reflects an estimate of the
    true value 1 hour prior rather than whatever the previous *row* happened to be."""
    continuous = (
        df.set_index("date_time")[[TARGET_COLUMN]]
        .reindex(pd.date_range(df["date_time"].min(), df["date_time"].max(), freq="h"))
    )
    continuous[TARGET_COLUMN] = continuous[TARGET_COLUMN].interpolate(method="linear")

    lags = pd.DataFrame(index=continuous.index)
    lags["lag_1h"] = continuous[TARGET_COLUMN].shift(1)
    lags["lag_3h"] = continuous[TARGET_COLUMN].shift(3)
    lags["lag_24h"] = continuous[TARGET_COLUMN].shift(24)
    lags["rolling_mean_3h"] = continuous[TARGET_COLUMN].shift(1).rolling(3).mean()

    df = df.merge(lags, left_on="date_time", right_index=True, how="left")
    return df.dropna(subset=LAG_COLUMNS).reset_index(drop=True)


def congestion_label(volume: pd.Series) -> pd.Series:
    """Bin raw traffic volume into 4 congestion levels using quartiles of the training data."""
    bins = volume.quantile([0, 0.25, 0.5, 0.75, 1.0]).to_numpy().copy()
    bins[0] -= 1
    bins[-1] += 1
    labels = ["Low", "Moderate", "High", "Severe"]
    return pd.cut(volume, bins=bins, labels=labels)


def load_dataset(path: str = DATA_PATH):
    df = add_lag_features(engineer_features(load_raw(path)))
    df["congestion_level"] = congestion_label(df[TARGET_COLUMN])
    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]
    return df, X, y


def seasonal_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Mean traffic volume by (hour, day_of_week), used as a fallback estimate for
    lag features at live-inference time when no real-time sensor reading is available
    (e.g. a camera-only deployment with no volume sensor at all)."""
    return df.groupby(["hour", "day_of_week"])[TARGET_COLUMN].mean().rename("seasonal_mean")


def train_val_test_split(df: pd.DataFrame, val_frac: float = 0.15, test_frac: float = 0.15):
    """Chronological split (no shuffling) since this is time-series data."""
    n = len(df)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)
    train = df.iloc[: n - n_val - n_test]
    val = df.iloc[n - n_val - n_test: n - n_test]
    test = df.iloc[n - n_test:]
    return train, val, test


if __name__ == "__main__":
    df, X, y = load_dataset()
    print(df[["date_time", *FEATURE_COLUMNS, "traffic_volume", "congestion_level"]].head())
    print(f"\nRows: {len(df)}  Date range: {df['date_time'].min()} -> {df['date_time'].max()}")
    print(f"\nCongestion level counts:\n{df['congestion_level'].value_counts()}")
