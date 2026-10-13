"""Tests for the UI theme helpers (pure string builders, no Streamlit)."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ui_theme import (  # noqa: E402
    HEADER_CROSS_SECONDS, LEVELS, STATUS_MESSAGES, badge_html, counter_html, header_html, status_message, theme_css,
)


def test_every_level_has_friendly_messages():
    for level in LEVELS:
        assert len(STATUS_MESSAGES[level]) >= 3
        assert status_message(level, "cam") in STATUS_MESSAGES[level]


def test_status_message_is_stable_for_the_same_key_and_unknown_level_is_blank():
    assert status_message("High", "I-5 Slauson") == status_message("High", "I-5 Slauson")
    assert status_message("Nonsense") == ""


def test_status_message_varies_with_key():
    picks = {status_message("Low", f"cam{i}") for i in range(20)}
    assert len(picks) > 1


def test_header_cars_slow_down_as_congestion_rises():
    assert HEADER_CROSS_SECONDS["Low"] < HEADER_CROSS_SECONDS["Moderate"] < HEADER_CROSS_SECONDS["High"] \
        < HEADER_CROSS_SECONDS["Severe"]
    assert "--tc-dur:7s" in header_html("T", "s", "Low")
    assert "--tc-dur:40s" in header_html("T", "s", "Severe")
    assert "--tc-dur:10s" in header_html("T", "s", None)  # no reading yet -> default speed


def test_badge_pulses_only_for_high_and_severe():
    assert "tc-pulse" not in badge_html("Low", "#22c55e")
    assert "tc-pulse" not in badge_html("Moderate", "#eab308")
    assert "tc-pulse" in badge_html("High", "#f97316")
    assert "tc-pulse" in badge_html("Severe", "#ef4444")


def test_badge_shows_level_and_optional_message():
    html = badge_html("Moderate", "#eab308", message="Busy but moving.")
    assert "Moderate" in html and "Busy but moving." in html
    assert "tc-badge-msg" not in badge_html("Moderate", "#eab308")


def test_counter_keeps_the_real_value_readable_and_never_goes_negative():
    html = counter_html(37, "Vehicle count")
    assert "--tc-n:37" in html and 'class="tc-sr">37<' in html and "Vehicle count" in html
    assert "--tc-n:0" in counter_html(-5, "x")


def test_css_respects_reduced_motion():
    css = theme_css()
    assert "prefers-reduced-motion" in css and "@property --tc-n" in css
