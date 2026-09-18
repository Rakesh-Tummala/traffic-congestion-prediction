"""Tests for the downloadable PNG report generator."""
import io
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from report import PANEL_HEIGHT, build_report_png  # noqa: E402


def _sample_frame(w=200, h=100):
    return np.zeros((h, w, 3), dtype=np.uint8)


def test_build_report_png_returns_valid_png_bytes():
    png_bytes = build_report_png(_sample_frame(), "Test Camera", "Moderate", 5, 0.1234, "LSTM")
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"  # PNG file signature


def test_build_report_png_adds_panel_below_frame():
    w, h = 200, 100
    png_bytes = build_report_png(_sample_frame(w, h), "Test Camera", "Low", 0, 0.0, "SVR")
    img = Image.open(io.BytesIO(png_bytes))
    assert img.size == (w, h + PANEL_HEIGHT)


def test_build_report_png_works_for_every_congestion_level():
    for level in ("Low", "Moderate", "High", "Severe"):
        png_bytes = build_report_png(_sample_frame(), "Cam", level, 1, 0.05, "LSTM")
        assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_report_png_handles_unknown_level_without_crashing():
    png_bytes = build_report_png(_sample_frame(), "Cam", "Unknown", 1, 0.05, "LSTM")
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"


def test_build_report_png_falls_back_when_arial_unavailable(monkeypatch):
    # Simulates the realistic case (e.g. Linux CI without Arial installed):
    # the specific "arial.ttf" lookup fails, so _load_fonts() should fall
    # back to Pillow's own bundled default font rather than crashing.
    import report as report_module
    real_truetype = report_module.ImageFont.truetype

    def _fail_only_for_arial(font, *args, **kwargs):
        if font == "arial.ttf":
            raise OSError("cannot open resource: arial.ttf")
        return real_truetype(font, *args, **kwargs)

    monkeypatch.setattr(report_module.ImageFont, "truetype", _fail_only_for_arial)
    png_bytes = build_report_png(_sample_frame(), "Cam", "High", 3, 0.2, "LSTM")
    assert png_bytes[:8] == b"\x89PNG\r\n\x1a\n"
