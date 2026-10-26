"""Small tests for the pure helpers in research/ (statistics and bucketing)."""
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
sys.path.insert(0, os.path.join(HERE, "..", "research"))
sys.path.insert(0, os.path.join(HERE, "..", "src"))

import label_agreement as la  # noqa: E402
import sample_for_labels as sfl  # noqa: E402
import violation_eval as ve  # noqa: E402
import baselines as bl  # noqa: E402


def test_icc_is_one_for_identical_counts_and_low_for_noise():
    x = np.arange(30, dtype=float)
    assert la.icc_2_1(x, x) > 0.999
    rng = np.random.default_rng(0)
    assert la.icc_2_1(x, rng.permutation(x)) < 0.3


def test_agreement_reports_a_known_bias():
    a = pd.DataFrame({"id": list("abcdef"), "count": [1, 5, 9, 14, 22, 30], "usable": "yes"})
    b = a.assign(count=a["count"] + 2)
    r = la.agreement(a, b)
    assert abs(r["bias_a_minus_b"] + 2) < 1e-9 and r["mean_abs_diff"] == 2 and r["pearson_r"] > 0.99


def test_time_buckets_cover_every_hour_exactly_once():
    names = [sfl.bucket_of(h) for h in range(24)]
    assert "other" not in names
    assert names[3] == "night" and names[6] == "dawn" and names[8] == "morning" and names[12] == "midday" \
        and names[17] == "afternoon" and names[20] == "dusk" and names[23] == "night"


def test_wilson_interval_contains_the_estimate_and_is_wide_for_small_n():
    lo, hi = ve.wilson(8, 10)
    assert lo < 0.8 < hi and hi - lo > 0.3
    lo2, hi2 = ve.wilson(800, 1000)
    assert hi2 - lo2 < 0.06
    assert all(np.isnan(v) for v in ve.wilson(0, 0))


def test_block_bootstrap_indices_stay_in_range_and_keep_blocks_contiguous():
    idx = bl.block_bootstrap_idx(100, 24, 5, np.random.default_rng(0))
    assert idx.shape == (5, 100) and idx.min() >= 0 and idx.max() < 100
    assert (np.diff(idx[:, :24], axis=1) == 1).all()


def test_level_scores_use_training_edges():
    edges = np.array([1000, 3000, 5000])
    s = bl.level_scores(np.array([500, 2000, 4000, 6000]), np.array([600, 3500, 4500, 5500]), edges)
    assert s["LevelAcc"] == 0.75 and s["Within1"] == 1.0
