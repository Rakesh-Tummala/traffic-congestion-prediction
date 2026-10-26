"""Inter-annotator agreement between two people's hand counts of the same frames.

A paper that scores a detector against hand counts has to show that the hand counts themselves are reliable.
Two annotators count the same frames independently (research/sample_for_labels.py makes the sheets); this reports
the mean absolute difference, bias, Bland-Altman 95% limits of agreement, Pearson r, and the intraclass
correlation ICC(2,1) with the usual two-way random-effects formula.

    python research/label_agreement.py --a research/data/labels_new.csv --b research/data/labels_second.csv
"""
import argparse

import numpy as np
import pandas as pd


def icc_2_1(x: np.ndarray, y: np.ndarray) -> float:
    data = np.c_[x, y].astype(float)
    n, k = data.shape
    grand = data.mean()
    ms_rows = k * ((data.mean(axis=1) - grand) ** 2).sum() / (n - 1)
    ms_cols = n * ((data.mean(axis=0) - grand) ** 2).sum() / (k - 1)
    sse = ((data - data.mean(axis=1, keepdims=True) - data.mean(axis=0, keepdims=True) + grand) ** 2).sum()
    ms_err = sse / ((n - 1) * (k - 1))
    return float((ms_rows - ms_err) / (ms_rows + (k - 1) * ms_err + k * (ms_cols - ms_err) / n))


def agreement(a: pd.DataFrame, b: pd.DataFrame) -> dict:
    m = a.merge(b, on="id", suffixes=("_a", "_b"))
    m = m[(m["usable_a"].astype(str).str.lower() == "yes") & (m["usable_b"].astype(str).str.lower() == "yes")]
    m = m.dropna(subset=["count_a", "count_b"])
    d = m["count_a"].astype(float) - m["count_b"].astype(float)
    sd = d.std(ddof=1)
    return {"n_frames": len(m), "mean_abs_diff": float(d.abs().mean()), "bias_a_minus_b": float(d.mean()),
            "limits_low": float(d.mean() - 1.96 * sd), "limits_high": float(d.mean() + 1.96 * sd),
            "pearson_r": float(np.corrcoef(m["count_a"], m["count_b"])[0, 1]),
            "ICC_2_1": icc_2_1(m["count_a"].to_numpy(), m["count_b"].to_numpy())}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    args = ap.parse_args()
    for k, v in agreement(pd.read_csv(args.a), pd.read_csv(args.b)).items():
        print(f"{k:16s} {v:.3f}" if isinstance(v, float) else f"{k:16s} {v}")


if __name__ == "__main__":
    main()
