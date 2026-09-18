"""Tests for the local prediction history log."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import prediction_log  # noqa: E402
from prediction_log import load_history, log_prediction  # noqa: E402


@pytest.fixture(autouse=True)
def _isolated_log(tmp_path, monkeypatch):
    monkeypatch.setattr(prediction_log, "LOG_PATH", str(tmp_path / "prediction_log.json"))
    yield


def test_log_prediction_roundtrips():
    log_prediction("Cam A", "lstm", 2500.0, "High", "Moderate", "High")
    history = load_history("Cam A")
    assert len(history) == 1
    assert history[0]["camera_name"] == "Cam A"
    assert history[0]["final_label"] == "High"
    assert history[0]["volume"] == 2500.0


def test_load_history_filters_by_camera():
    log_prediction("Cam A", "lstm", 1000.0, "Low", "Low", "Low")
    log_prediction("Cam B", "lstm", 3000.0, "Severe", "Severe", "Severe")
    log_prediction("Cam A", "svr", 1200.0, "Low", "Moderate", "Moderate")

    assert len(load_history("Cam A")) == 2
    assert len(load_history("Cam B")) == 1
    assert len(load_history()) == 3


def test_load_history_respects_limit_and_returns_most_recent():
    for i in range(5):
        log_prediction("Cam A", "lstm", float(i), "Low", "Low", "Low")
    recent = load_history("Cam A", limit=2)
    assert len(recent) == 2
    assert [e["volume"] for e in recent] == [3.0, 4.0]


def test_log_trims_to_max_entries(monkeypatch):
    monkeypatch.setattr(prediction_log, "MAX_LOG_ENTRIES", 3)
    for i in range(5):
        log_prediction("Cam A", "lstm", float(i), "Low", "Low", "Low")
    full_history = load_history("Cam A", limit=100)
    assert len(full_history) == 3
    assert [e["volume"] for e in full_history] == [2.0, 3.0, 4.0]
