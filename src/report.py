"""Compose a small shareable PNG summary of a single congestion prediction:
the annotated camera frame with a text panel appended below it (camera name,
congestion level, vehicle count, model, timestamp). Lets a user share or save
one specific result without needing to open the dashboard."""
import io
from datetime import datetime

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

LEVEL_COLORS_RGB = {
    "Low": (34, 197, 94), "Moderate": (234, 179, 8),
    "High": (249, 115, 22), "Severe": (239, 68, 68),
}
PANEL_HEIGHT = 140
BACKGROUND_RGB = (17, 24, 39)
TEXT_RGB = (229, 231, 235)


def _load_fonts():
    """Falls back to Pillow's own bundled default font if Arial isn't
    installed on this system (e.g. a minimal Linux CI image) — the report
    still generates, just with a plainer font."""
    try:
        return ImageFont.truetype("arial.ttf", 32), ImageFont.truetype("arial.ttf", 18)
    except Exception:
        default = ImageFont.load_default()
        return default, default


def build_report_png(annotated_frame_bgr: np.ndarray, camera_name: str, congestion_level: str,
                      vehicle_count: int, density_ratio: float, model_name: str) -> bytes:
    """Returns PNG-encoded bytes of the annotated frame plus a summary panel."""
    frame_rgb = cv2.cvtColor(annotated_frame_bgr, cv2.COLOR_BGR2RGB)
    frame_img = Image.fromarray(frame_rgb)
    w, h = frame_img.size

    canvas = Image.new("RGB", (w, h + PANEL_HEIGHT), BACKGROUND_RGB)
    canvas.paste(frame_img, (0, 0))

    draw = ImageDraw.Draw(canvas)
    font_big, font_small = _load_fonts()
    color = LEVEL_COLORS_RGB.get(congestion_level, (200, 200, 200))

    draw.rectangle([20, h + 15, 240, h + 65], outline=color, width=3)
    draw.text((30, h + 22), congestion_level.upper(), font=font_big, fill=color)

    info_lines = [
        camera_name,
        f"{vehicle_count} vehicles detected  ·  density {density_ratio:.4f}  ·  model: {model_name}",
        f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} — Smart Traffic Congestion Prediction",
    ]
    for i, line in enumerate(info_lines):
        draw.text((270, h + 15 + i * 24), line, font=font_small, fill=TEXT_RGB)

    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()
