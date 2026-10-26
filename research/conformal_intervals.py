"""Split-conformal prediction intervals for the DEPLOYED models (the files in models/), written to
models/conformal_intervals.json and used by the app's forecast chart.

For each model: |residual| on the validation rows gives the half-width q such that a new hour's true volume
falls inside [prediction - q, prediction + q] with at least the nominal probability (split conformal, which
assumes the validation and test hours are exchangeable; the coverage actually achieved on the later test
rows is reported so any drift shows up). Run after training:

    python research/conformal_intervals.py
"""
import json
import os
import sys

import joblib
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

from data_loader import FEATURE_COLUMNS, TARGET_COLUMN, load_dataset, train_val_test_split  # noqa: E402
from train_dl import SEQ_LEN, TrafficLSTM  # noqa: E402

MODELS = os.path.join(HERE, "..", "models")


def tabular_predict(name, df):
    bundle = joblib.load(os.path.join(MODELS, f"{name}.joblib"))
    X = bundle["scaler"].transform(df[bundle["features"]])
    return bundle["model"].predict(X)


def lstm_predict(df):
    sc = joblib.load(os.path.join(MODELS, "lstm_scalers.joblib"))
    model = TrafficLSTM(n_features=len(sc["features"]))
    model.load_state_dict(torch.load(os.path.join(MODELS, "lstm.pt"), map_location="cpu"))
    model.eval()
    X = sc["x_scaler"].transform(df[sc["features"]]).astype(np.float32)
    seqs = np.stack([X[i: i + SEQ_LEN] for i in range(len(X) - SEQ_LEN)]).astype(np.float32)
    with torch.no_grad():
        p = model(torch.from_numpy(seqs)).numpy()
    return sc["y_scaler"].inverse_transform(p.reshape(-1, 1)).ravel()  # aligned with df.iloc[SEQ_LEN:]


def main():
    df, _, _ = load_dataset()
    train, val, test = train_val_test_split(df)
    out = {}
    report = []
    for key, getter in (("linear_regression", lambda d: tabular_predict("linear_regression", d)),
                        ("svr", lambda d: tabular_predict("svr", d)),
                        ("gradient_boosting", lambda d: tabular_predict("gradient_boosting", d)),
                        ("lstm", lstm_predict)):
        pv, pt = getter(val), getter(test)
        yv, yt = val[TARGET_COLUMN].to_numpy(), test[TARGET_COLUMN].to_numpy()
        if key == "lstm":
            yv, yt = yv[SEQ_LEN:], yt[SEQ_LEN:]
        res = np.abs(yv - pv)
        n = len(res)
        out[key] = {}
        for alpha in (0.2, 0.1):
            q = float(np.quantile(res, min(1.0, np.ceil((n + 1) * (1 - alpha)) / n), method="higher"))
            cover = float(np.mean(np.abs(yt - pt) <= q))
            out[key][f"{int((1 - alpha) * 100)}"] = {"half_width": round(q, 1), "test_coverage": round(cover, 3)}
            report.append(f"{key:18s} nominal {1 - alpha:.0%}  half-width {q:7.1f}  coverage on later test hours {cover:.1%}")
    json.dump({"note": "split-conformal half-widths in vehicles/hour from validation residuals of the deployed models",
               "intervals": out}, open(os.path.join(MODELS, "conformal_intervals.json"), "w"), indent=2)
    print("\n".join(report))


if __name__ == "__main__":
    main()
