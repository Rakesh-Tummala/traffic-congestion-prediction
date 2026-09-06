"""Tests for the ANPR module. The real OCR test downloads model weights on
first run (slow, one-time, needs network) — it's the one test in this suite
that isn't instant, kept to a single case since the reader is cached."""
import os
import sys

import cv2
import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from anpr import read_plate  # noqa: E402


def test_read_plate_empty_box_returns_none():
    frame = np.zeros((100, 100, 3), dtype=np.uint8)
    result = read_plate(frame, box=(50, 50, 50, 50))  # zero-area box
    assert result["text"] is None
    assert result["confidence"] == 0.0


def test_read_plate_blank_frame_returns_none():
    frame = np.full((200, 200, 3), 127, dtype=np.uint8)  # flat gray, no text
    result = read_plate(frame, box=(0, 0, 200, 200))
    assert result["text"] is None


@pytest.mark.slow
def test_read_plate_reads_clear_synthetic_text():
    """Sanity check that the OCR mechanism itself works on unambiguous text —
    proves the pipeline is wired correctly, independent of how well real
    camera footage's blurry/occluded plates read (documented separately)."""
    img = np.full((150, 400, 3), 255, dtype=np.uint8)
    cv2.putText(img, "KA01AB1234", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 4)
    result = read_plate(img, box=(0, 0, 400, 150))
    assert result["text"] is not None
    assert len(result["text"]) == len("KA01AB1234")
    # OCR on plate-style fonts commonly confuses visually similar glyphs
    # (0/O, 1/I) — normalize those before comparing rather than asserting an
    # exact match, since that confusion is a known, generic OCR limitation
    # and not something this code controls.
    normalize = str.maketrans({"O": "0", "I": "1"})
    assert result["text"].translate(normalize) == "KA01AB1234".translate(normalize)
    assert result["confidence"] > 0.3
