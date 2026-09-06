"""Attempt automatic license plate recognition (ANPR) on a detected vehicle's
cropped region, using EasyOCR (pure-Python, no external OCR binary needed).

Real-world caveat, confirmed by direct testing during development: plates are
frequently illegible or not visible at all in the vehicle crops this project's
detector produces — nighttime camera footage is too blurry/low-light, and
even in clear daytime traffic photos, most vehicles' rear plates are occluded
by neighboring vehicles or captured from an angle that doesn't show the plate
at all. A synthetic clear-text sanity check confirms the OCR mechanism itself
works correctly (reads clean text, occasionally confusing visually similar
characters like 0/O — a known, generic OCR limitation, not a bug in this
code) — so "not readable" on real footage should be treated as the expected
common case, not a malfunction. See tests/test_anpr.py.
"""
import re

import cv2
import numpy as np

_reader = None


def get_reader():
    """Lazily construct the EasyOCR reader — this downloads model weights on
    first use and is slow to initialize, so it's built once and reused."""
    global _reader
    if _reader is None:
        import easyocr
        _reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    return _reader


def read_plate(frame: np.ndarray, box: tuple, min_confidence: float = 0.35, upscale: int = 4) -> dict:
    """Try to read a license plate from within `box` (x1, y1, x2, y2) of
    `frame`. Returns {"text": str|None, "confidence": float} — text is None
    when nothing plate-like and confident enough was found (the common case
    on this project's real footage; see module docstring)."""
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return {"text": None, "confidence": 0.0}

    crop = frame[y1:y2, x1:x2]
    crop = cv2.resize(crop, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)

    results = get_reader().readtext(crop)

    best_text, best_conf = None, 0.0
    for _, text, conf in results:
        cleaned = re.sub(r"[^A-Za-z0-9]", "", text).upper()
        # Plates are short alphanumeric strings; skip obviously-not-a-plate
        # matches (too short/long, e.g. stray reflections or bumper stickers)
        # rather than returning noise as if it were a real reading.
        if 4 <= len(cleaned) <= 12 and conf > best_conf:
            best_text, best_conf = cleaned, conf

    if best_text is None or best_conf < min_confidence:
        return {"text": None, "confidence": round(float(best_conf), 3)}
    return {"text": best_text, "confidence": round(float(best_conf), 3)}
