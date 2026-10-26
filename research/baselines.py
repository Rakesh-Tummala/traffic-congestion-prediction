"""Research evaluation of the historical (time-series) models.

Answers the questions a reviewer will ask of the 0.977 R2 figure:

  1. How does it compare with trivial baselines ("same as an hour ago") and other model families?
  2. How much of the accuracy comes from the lag features, the calendar, the weather?
  3. How does the error grow with the forecast horizon?
  4. Is the ranking stable under expanding-window time-series cross-validation?
  5. Are the differences larger than the noise (moving-block bootstrap confidence intervals)?
  6. Do split-conformal prediction intervals achieve their nominal coverage?

Everything is seeded and writes CSV/Markdown/PNG files to research/results/.
Run:  python research/baselines.py            (about 15-25 minutes on a laptop CPU)
      python research/baselines.py --quick    (one seed, fewer folds; a few minutes)
"""
import argparse
import os
import sys
import time

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from data_loader import (FEATURE_COLUMNS, LAG_COLUMNS, TARGET_COLUMN, load_dataset,  # noqa: E402
                         train_val_test_split)
from train_dl import SEQ_LEN, TrafficLSTM, make_sequences  # noqa: E402

RESULTS = os.path.join(HERE, "results")
SEED = 42
LEVELS = ["Low", "Moderate", "High", "Severe"]

TIME_COLS = ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos", "is_weekend", "is_holiday"]
WEATHER_COLS = ["temp_c", "rain_1h", "snow_1h", "clouds_all", "weather_code"]


# ------------------------------------------------------------------ helpers
def metrics(y, p):
    return {"MAE": mean_absolute_error(y, p), "RMSE": float(np.sqrt(mean_squared_error(y, p))), "R2": r2_score(y, p)}


def level_labels(volume, edges):
    return np.digitize(volume, edges)  # 0..3


def level_scores(y, p, edges):
    ty, tp = level_labels(y, edges), level_labels(p, edges)
    return {"LevelAcc": float(np.mean(ty == tp)), "Within1": float(np.mean(np.abs(ty - tp) <= 1)),
            "MacroF1": float(f1_score(ty, tp, average="macro"))}


def block_bootstrap_idx(n, block, n_boot, rng):
    starts = rng.integers(0, n - block + 1, size=(n_boot, int(np.ceil(n / block))))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]).reshape(n_boot, -1)[:, :n]
    return idx


def add_extra_columns(df):
    """lag_168h (same hour last week) and the hour-by-weekday profile of the *training* rows."""
    cont = df.set_index("date_time")[TARGET_COLUMN]
    cont = cont.reindex(pd.date_range(cont.index.min(), cont.index.max(), freq="h")).interpolate(method="linear")
    df = df.copy()
    df["lag_168h"] = cont.shift(168).reindex(df["date_time"]).to_numpy()
    return df


class TrafficGRU(nn.Module):
    def __init__(self, n_features, hidden_size=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.rnn = nn.GRU(n_features, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.head = nn.Sequential(nn.Linear(hidden_size, 32), nn.ReLU(), nn.Linear(32, 1))

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :]).squeeze(-1)


SEQ_EPOCHS = 40


def train_sequence_model(kind, seed, train, val, test, epochs=None, patience=5, batch=128, lr=1e-3):
    """Same protocol as src/train_dl.py (24 h window, Adam, MSE, early stopping on validation)."""
    epochs = epochs or SEQ_EPOCHS
    torch.manual_seed(seed)
    np.random.seed(seed)
    xs = StandardScaler().fit(train[FEATURE_COLUMNS])
    ys = StandardScaler().fit(train[[TARGET_COLUMN]])

    def prep(split):
        X = xs.transform(split[FEATURE_COLUMNS]).astype(np.float32)
        y = ys.transform(split[[TARGET_COLUMN]]).astype(np.float32).ravel()
        return make_sequences(X, y, SEQ_LEN)

    Xtr, ytr = prep(train)
    Xva, yva = prep(val)
    Xte, _ = prep(test)
    model = (TrafficLSTM if kind == "LSTM" else TrafficGRU)(len(FEATURE_COLUMNS))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    gen = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)), batch_size=batch, shuffle=True, generator=gen)
    Xva_t, yva_t = torch.from_numpy(Xva), torch.from_numpy(yva)
    best, best_state, bad = float("inf"), None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva_t), yva_t).item()
        if v < best:
            best, best_state, bad = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pv = model(Xva_t).numpy()
        pt = model(torch.from_numpy(Xte)).numpy()
    inv = lambda a: ys.inverse_transform(a.reshape(-1, 1)).ravel()  # noqa: E731
    return inv(pv), inv(pt)  # predictions for val[SEQ_LEN:] and test[SEQ_LEN:]


def fit_predict_tabular(name, Xtr, ytr, Xte, seed=SEED):
    """Fit a tabular model on (scaled) features and predict."""
    if name == "LinearRegression":
        sc = StandardScaler().fit(Xtr)
        return LinearRegression().fit(sc.transform(Xtr), ytr).predict(sc.transform(Xte))
    if name == "SVR":
        sc = StandardScaler().fit(Xtr)
        n = min(8000, len(Xtr))
        idx = np.random.RandomState(seed).choice(len(Xtr), n, replace=False)
        m = SVR(kernel="rbf", C=100, epsilon=10, gamma="auto").fit(sc.transform(Xtr)[idx], np.asarray(ytr)[idx])
        return m.predict(sc.transform(Xte))
    if name == "RandomForest":
        m = RandomForestRegressor(n_estimators=200, min_samples_leaf=3, n_jobs=-1, random_state=seed)
        return m.fit(Xtr, ytr).predict(Xte)
    if name == "GradientBoosting":
        m = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, random_state=seed)
        return m.fit(Xtr, ytr).predict(Xte)
    raise ValueError(name)


TABULAR = ["LinearRegression", "SVR", "RandomForest", "GradientBoosting"]


# ------------------------------------------------------------------ experiments
def main_comparison(train, val, test, edges, seeds, log):
    """All models on the same test rows (the LSTM/GRU cannot predict the first SEQ_LEN rows)."""
    ytest = test[TARGET_COLUMN].to_numpy()[SEQ_LEN:]
    preds = {}
    t = test.iloc[SEQ_LEN:]
    preds["Persistence (last hour)"] = t["lag_1h"].to_numpy()
    preds["Seasonal naive (24 h ago)"] = t["lag_24h"].to_numpy()
    preds["Seasonal naive (1 week ago)"] = t["lag_168h"].to_numpy()
    prof = train.groupby(["hour", "day_of_week"])[TARGET_COLUMN].mean()
    preds["Hour x weekday mean"] = np.array([prof.loc[(h, d)] for h, d in zip(t["hour"], t["day_of_week"])])
    for name in TABULAR:
        log(f"  fitting {name}")
        preds[name] = fit_predict_tabular(name, train[FEATURE_COLUMNS], train[TARGET_COLUMN], test[FEATURE_COLUMNS])[SEQ_LEN:]
    val_preds = {}
    seq_preds = {}
    for kind in ("LSTM", "GRU"):
        runs = []
        for s in seeds:
            log(f"  training {kind} seed {s}")
            pv, pt = train_sequence_model(kind, s, train, val, test)
            runs.append((pv, pt))
        seq_preds[kind] = runs
        preds[f"{kind} (mean of {len(seeds)} seeds)"] = np.mean([r[1] for r in runs], axis=0)
        val_preds[kind] = np.mean([r[0] for r in runs], axis=0)
    rows = []
    for name, p in preds.items():
        rows.append({"Model": name, **metrics(ytest, p), **level_scores(ytest, p, edges)})
    # seed spread for the recurrent models
    spread = []
    for kind, runs in seq_preds.items():
        per_seed = [metrics(ytest, r[1]) for r in runs]
        spread.append({"Model": kind, "seeds": len(runs),
                       "MAE_mean": np.mean([m["MAE"] for m in per_seed]), "MAE_std": np.std([m["MAE"] for m in per_seed]),
                       "R2_mean": np.mean([m["R2"] for m in per_seed]), "R2_std": np.std([m["R2"] for m in per_seed])})
    return pd.DataFrame(rows), pd.DataFrame(spread), preds, ytest, val_preds


def ablation(train, test, edges, log):
    groups = {
        "All features": TIME_COLS + WEATHER_COLS + LAG_COLUMNS,
        "No lag features": TIME_COLS + WEATHER_COLS,
        "Lag features only": LAG_COLUMNS,
        "Calendar only": TIME_COLS,
        "Weather only": WEATHER_COLS,
        "No weather": TIME_COLS + LAG_COLUMNS,
        "No calendar": WEATHER_COLS + LAG_COLUMNS,
    }
    rows = []
    for gname, cols in groups.items():
        for m in ("LinearRegression", "GradientBoosting", "SVR"):
            log(f"  ablation {gname} / {m}")
            p = fit_predict_tabular(m, train[cols], train[TARGET_COLUMN], test[cols])
            rows.append({"Features": gname, "Model": m, "n_features": len(cols), **metrics(test[TARGET_COLUMN].to_numpy(), p)})
    return pd.DataFrame(rows)


def horizon_experiment(df, edges, log):
    """Predict y(t+h) from information available at t: y(t), y(t-2), y(t-23), their 3 h mean,
    the calendar of the target hour, and the weather at t (persisted)."""
    cont = df.set_index("date_time")[TARGET_COLUMN]
    full_idx = pd.date_range(cont.index.min(), cont.index.max(), freq="h")
    observed = pd.Series(False, index=full_idx)
    observed.loc[cont.index] = True
    # the sensor has one gap of ~7,400 hours; interpolating across it would invent traffic, so only short gaps
    # (6 h or less) are bridged and every target must itself be a real observation
    cont = cont.reindex(full_idx).interpolate(method="linear", limit=6, limit_area="inside")
    rows = []
    for h in (1, 3, 6, 12, 24):
        base = pd.DataFrame({"y0": cont, "y2": cont.shift(2), "y23": cont.shift(23)})
        base["m3"] = (cont + cont.shift(1) + cont.shift(2)) / 3
        base["target"] = cont.shift(-h).where(observed.shift(-h, fill_value=False) & observed)
        tt = base.index + pd.Timedelta(hours=h)
        base["hour_sin"], base["hour_cos"] = np.sin(2 * np.pi * tt.hour / 24), np.cos(2 * np.pi * tt.hour / 24)
        base["dow_sin"], base["dow_cos"] = np.sin(2 * np.pi * tt.dayofweek / 7), np.cos(2 * np.pi * tt.dayofweek / 7)
        base["month_sin"], base["month_cos"] = np.sin(2 * np.pi * tt.month / 12), np.cos(2 * np.pi * tt.month / 12)
        base["is_weekend"] = (tt.dayofweek >= 5).astype(int)
        base["seasonal24"] = cont.shift(24 - h)  # y(t+h-24)
        w = df.set_index("date_time")[["temp_c", "rain_1h", "snow_1h", "clouds_all", "weather_code", "is_holiday"]]
        w = w.reindex(full_idx).ffill().bfill()
        base = base.join(w).dropna()
        n = len(base)
        n_test, n_val = int(n * 0.15), int(n * 0.15)
        tr, te = base.iloc[: n - n_val - n_test], base.iloc[n - n_test:]
        feats = ["y0", "y2", "y23", "m3", "hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos",
                 "is_weekend", "temp_c", "rain_1h", "snow_1h", "clouds_all", "weather_code", "is_holiday"]
        yt = te["target"].to_numpy()
        prof = tr.assign(hour=np.round(np.degrees(np.arctan2(tr.hour_sin, tr.hour_cos)) / 15).astype(int) % 24,
                         dow=np.round(np.degrees(np.arctan2(tr.dow_sin, tr.dow_cos)) / (360 / 7)).astype(int) % 7)
        prof_mean = prof.groupby(["hour", "dow"])["target"].mean()
        te_h = np.round(np.degrees(np.arctan2(te.hour_sin, te.hour_cos)) / 15).astype(int) % 24
        te_d = np.round(np.degrees(np.arctan2(te.dow_sin, te.dow_cos)) / (360 / 7)).astype(int) % 7
        preds = {"Persistence": te["y0"].to_numpy(), "Seasonal naive (24 h)": te["seasonal24"].to_numpy(),
                 "Hour x weekday mean": np.array([prof_mean.get((a, b), np.nan) for a, b in zip(te_h, te_d)])}
        for m in ("LinearRegression", "GradientBoosting"):
            log(f"  horizon {h} h / {m}")
            preds[m] = fit_predict_tabular(m, tr[feats], tr["target"], te[feats])
        for name, p in preds.items():
            ok = ~np.isnan(p)
            rows.append({"Horizon_h": h, "Model": name, **metrics(yt[ok], p[ok])})
    return pd.DataFrame(rows)


def tscv_experiment(df, n_splits, log):
    use = df.iloc[: int(len(df) * 0.85)]
    tss = TimeSeriesSplit(n_splits=n_splits)
    rows = []
    for k, (a, b) in enumerate(tss.split(use)):
        tr, te = use.iloc[a], use.iloc[b]
        folds = {"Persistence": te["lag_1h"].to_numpy(), "Seasonal naive (24 h)": te["lag_24h"].to_numpy()}
        for name in TABULAR:
            log(f"  tscv fold {k + 1}/{n_splits} {name}")
            folds[name] = fit_predict_tabular(name, tr[FEATURE_COLUMNS], tr[TARGET_COLUMN], te[FEATURE_COLUMNS])
        for name, p in folds.items():
            rows.append({"Fold": k + 1, "train_rows": len(tr), "Model": name, **metrics(te[TARGET_COLUMN].to_numpy(), p)})
    raw = pd.DataFrame(rows)
    agg = raw.groupby("Model").agg(MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"), RMSE_mean=("RMSE", "mean"),
                                   R2_mean=("R2", "mean"), R2_std=("R2", "std")).reset_index()
    return raw, agg


def bootstrap_experiment(preds, ytest, n_boot=2000):
    rng = np.random.default_rng(SEED)
    idx = block_bootstrap_idx(len(ytest), 24, n_boot, rng)
    err = {k: np.abs(ytest - v) for k, v in preds.items()}
    ref = "Persistence (last hour)"
    rows = []
    for name, e in err.items():
        mae_b = e[idx].mean(axis=1)
        lo, hi = np.percentile(mae_b, [2.5, 97.5])
        row = {"Model": name, "MAE": e.mean(), "MAE_CI_low": lo, "MAE_CI_high": hi}
        if name != ref:
            d = (e - err[ref])[idx].mean(axis=1)
            dl, dh = np.percentile(d, [2.5, 97.5])
            row.update({"dMAE_vs_persistence": (e - err[ref]).mean(), "dMAE_CI_low": dl, "dMAE_CI_high": dh,
                        "significant": bool(dh < 0 or dl > 0)})
        rows.append(row)
    return pd.DataFrame(rows)


def conformal_experiment(train, val, test, val_preds, test_preds, ytest):
    """Split-conformal intervals: q = ceil((n+1)(1-alpha))/n quantile of |residual| on the validation set."""
    yval_tab = val[TARGET_COLUMN].to_numpy()
    rows, saved = [], {}
    cand = {}
    for name in ("LinearRegression", "GradientBoosting"):
        pv = fit_predict_tabular(name, train[FEATURE_COLUMNS], train[TARGET_COLUMN], val[FEATURE_COLUMNS])
        cand[name] = (yval_tab, pv, test_preds[name])
    yval_seq = val[TARGET_COLUMN].to_numpy()[SEQ_LEN:]
    cand["LSTM"] = (yval_seq, val_preds["LSTM"], test_preds[[k for k in test_preds if k.startswith("LSTM")][0]])
    for name, (yv, pv, pt) in cand.items():
        res = np.abs(yv - pv)
        n = len(res)
        for alpha in (0.2, 0.1):
            q = float(np.quantile(res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
            cover = float(np.mean((ytest >= pt - q) & (ytest <= pt + q)))
            rows.append({"Model": name, "nominal": 1 - alpha, "half_width": q, "test_coverage": cover})
            saved.setdefault(name, {})[f"{int((1 - alpha) * 100)}"] = q
    return pd.DataFrame(rows), saved


# ------------------------------------------------------------------ report
def md_table(df, floatfmt="{:.3f}"):
    cols = list(df.columns)
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, (float, np.floating)):
                cells.append(floatfmt.format(v) if abs(v) < 1000 else f"{v:,.0f}")
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(seeds, n_splits):
    r = lambda name: pd.read_csv(f"{RESULTS}/{name}.csv")  # noqa: E731
    md = ["# Historical-model research evaluation", "",
          f"Seeds: {seeds}. Chronological split 70/15/15; the test rows are the last 15% of the data. "
          "Congestion-level edges come from the training rows only. Recurrent-model rows are the average "
          "prediction of the seeds (an ensemble); the seed spread is listed separately.", "",
          "## 1. Main comparison (same test rows for every model)", "", md_table(r("main_comparison")), "",
          "Recurrent-model seed spread:", "", md_table(r("seed_spread")), "",
          "## 2. Moving-block bootstrap (24 h blocks, 2,000 resamples)", "", md_table(r("bootstrap")), "",
          "## 3. Feature ablation", "", md_table(r("ablation")), "",
          "## 4. Forecast horizon (targets are real observations only)", "", md_table(r("horizon")), "",
          f"## 5. Expanding-window time-series CV ({n_splits} folds, first 85% of the data)", "", md_table(r("tscv")), "",
          "## 6. Split-conformal prediction intervals (models retrained here; the app uses research/conformal_intervals.py)", "",
          md_table(r("conformal")), ""]
    open(f"{RESULTS}/results.md", "w", encoding="utf-8").write("\n".join(md))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="one seed, 3 CV folds, 2 epochs for the recurrent models (smoke test)")
    ap.add_argument("--horizon-only", action="store_true", help="rerun only the horizon experiment and rebuild results.md from the saved CSVs")
    args = ap.parse_args()
    global SEQ_EPOCHS
    if args.quick:
        SEQ_EPOCHS = 2
    os.makedirs(RESULTS, exist_ok=True)
    seeds = [42] if args.quick else [42, 43, 44]
    n_splits = 3 if args.quick else 5
    t0 = time.time()

    def log(msg):
        print(f"[{time.time() - t0:6.0f}s] {msg}", flush=True)

    np.random.seed(SEED)
    df, _, _ = load_dataset()
    df = add_extra_columns(df).dropna(subset=["lag_168h"]).reset_index(drop=True)
    train, val, test = train_val_test_split(df)
    edges = np.quantile(train[TARGET_COLUMN], [0.25, 0.5, 0.75])  # congestion bins from TRAINING rows only
    log(f"rows: train {len(train)}, val {len(val)}, test {len(test)}; level edges {np.round(edges)}")

    if args.horizon_only:
        log("horizon experiment only")
        hor_df = horizon_experiment(df, edges, log)
        hor_df.to_csv(f"{RESULTS}/horizon.csv", index=False)
        write_report(seeds, n_splits)
        log("done")
        return

    log("1/6 main comparison")
    main_df, spread_df, preds, ytest, val_preds = main_comparison(train, val, test, edges, seeds, log)
    main_df.to_csv(f"{RESULTS}/main_comparison.csv", index=False)
    spread_df.to_csv(f"{RESULTS}/seed_spread.csv", index=False)

    log("2/6 bootstrap confidence intervals")
    boot_df = bootstrap_experiment(preds, ytest)
    boot_df.to_csv(f"{RESULTS}/bootstrap.csv", index=False)

    log("3/6 ablation")
    abl_df = ablation(train, test, edges, log)
    abl_df.to_csv(f"{RESULTS}/ablation.csv", index=False)

    log("4/6 horizon experiment")
    hor_df = horizon_experiment(df, edges, log)
    hor_df.to_csv(f"{RESULTS}/horizon.csv", index=False)

    log("5/6 time-series cross-validation")
    tscv_raw, tscv_agg = tscv_experiment(df, n_splits, log)
    tscv_raw.to_csv(f"{RESULTS}/tscv_folds.csv", index=False)
    tscv_agg.to_csv(f"{RESULTS}/tscv.csv", index=False)

    log("6/6 conformal intervals")
    named_test = {k: v for k, v in preds.items()}
    conf_df, saved = conformal_experiment(train, val, test, val_preds, named_test, ytest)
    conf_df.to_csv(f"{RESULTS}/conformal.csv", index=False)
    import json
    json.dump({"note": "split-conformal half-widths (vehicles/hour) from validation residuals", "intervals": saved},
              open(f"{RESULTS}/conformal_baselines.json", "w"), indent=2)

    # figures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(5.2, 3.4), dpi=200)
    for m, g in hor_df.groupby("Model"):
        ax.plot(g["Horizon_h"], g["MAE"], marker="o", label=m)
    ax.set_xlabel("Forecast horizon (hours)")
    ax.set_ylabel("MAE (vehicles/hour)")
    ax.set_xscale("log")
    ax.set_xticks([1, 3, 6, 12, 24])
    ax.set_xticklabels(["1", "3", "6", "12", "24"])
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f"{RESULTS}/horizon_mae.png")

    write_report(seeds, n_splits)
    log("done")


if __name__ == "__main__":
    main()
