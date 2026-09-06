"""Best-effort vehicle color and body-type identification for a single
detected-vehicle crop. This is a demo feature, not a production classifier:

- Color is read as the dominant paint color from the crop, filtering out
  very dark pixels (shadows, tinted glass, tires) and very bright pixels
  (glare/reflections) before clustering — plain clustering on the raw crop
  tends to pick the vehicle's windows/tires instead of its body paint.
- Body type ("sedan", "SUV", "pickup truck", ...) comes from CLIP
  (openai/clip-vit-base-patch32), a general zero-shot image/text model run
  against a fixed list of body-style prompts. It was not trained for this
  and gives only moderate-confidence guesses on small, low-resolution
  traffic-camera crops (verified manually on real daytime crops: correct
  but ~30-55% confidence; worse on blurry/small nighttime crops) — good
  enough to label "what kind of car" for a demo, not for anything that
  needs to be reliably correct.
- There is deliberately no make/model (e.g. "Toyota Camry") identification:
  reading a badge or grille shape needs resolution these traffic cameras
  don't have, and guessing a specific make/model from body shape alone
  would be more misleading than useful.
"""
import cv2
import numpy as np

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
# (CLIP prompt, display label)
VEHICLE_TYPE_PROMPTS = [
    ("a sedan car", "sedan"),
    ("an SUV", "SUV"),
    ("a pickup truck", "pickup truck"),
    ("a hatchback car", "hatchback"),
    ("a minivan", "minivan"),
    ("a bus", "bus"),
    ("a semi truck", "truck"),
    ("a motorcycle", "motorcycle"),
]

_clip_model = None
_clip_processor = None


def get_clip():
    global _clip_model, _clip_processor
    if _clip_model is None:
        from transformers import CLIPModel, CLIPProcessor
        _clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
        _clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    return _clip_model, _clip_processor


def _crop(frame: np.ndarray, box):
    x1, y1, x2, y2 = (int(v) for v in box)
    h, w = frame.shape[:2]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    return frame[y1:y2, x1:x2]


def dominant_color_bgr(frame: np.ndarray, box, k: int = 3):
    """Return the dominant body-paint color of the vehicle in `box` as an
    (R, G, B) tuple, or None if the crop is empty/degenerate. Clusters only
    pixels in a mid brightness band so shadows/glass/tires (too dark) and
    glare/reflections (too bright) don't drown out the actual paint color."""
    crop = _crop(frame, box)
    if crop is None or crop.size == 0:
        return None

    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    v = hsv[:, :, 2]
    mask = (v > 60) & (v < 250)
    pixels = crop[mask]
    if pixels.shape[0] < 10:  # filter too aggressive for this crop — fall back to all pixels
        pixels = crop.reshape(-1, 3)
    if pixels.shape[0] < 1:
        return None

    pixels = np.float32(pixels)
    k = min(k, pixels.shape[0])
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5)
    _, labels, centers = cv2.kmeans(pixels, k, None, criteria, 5, cv2.KMEANS_RANDOM_CENTERS)
    counts = np.bincount(labels.flatten())
    b, g, r = (int(round(c)) for c in centers[np.argmax(counts)])
    return (r, g, b)


def color_name(rgb) -> str:
    """Map an (R, G, B) tuple to a human-readable color name via HSV thresholds."""
    r, g, b = rgb
    hsv = cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0][0]
    h, s, v = int(hsv[0]), int(hsv[1]), int(hsv[2])
    if v < 50:
        return "black"
    if s < 40:
        if v > 200:
            return "white"
        if v > 130:
            return "silver"
        return "gray"
    if h < 10 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 35:
        return "yellow"
    if h < 85:
        return "green"
    if h < 130:
        return "blue"
    return "purple"


def classify_vehicle_type(frame: np.ndarray, box, min_confidence: float = 0.25) -> dict:
    """Zero-shot body-style guess via CLIP. Returns
    {"type": str|None, "confidence": float}; type is None below `min_confidence`."""
    crop = _crop(frame, box)
    if crop is None or crop.size == 0:
        return {"type": None, "confidence": 0.0}

    from PIL import Image
    import torch

    model, processor = get_clip()
    image = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
    prompts = [p for p, _ in VEHICLE_TYPE_PROMPTS]
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = outputs.logits_per_image.softmax(dim=1)[0]
    best_idx = int(probs.argmax())
    confidence = float(probs[best_idx])
    if confidence < min_confidence:
        return {"type": None, "confidence": confidence}
    return {"type": VEHICLE_TYPE_PROMPTS[best_idx][1], "confidence": confidence}


def describe_vehicle(frame: np.ndarray, box) -> dict:
    """Combine color and body-type identification for one detected vehicle
    crop into a single result dict."""
    rgb = dominant_color_bgr(frame, box)
    type_result = classify_vehicle_type(frame, box)
    return {
        "color": color_name(rgb) if rgb is not None else None,
        "color_rgb": rgb,
        "type": type_result["type"],
        "type_confidence": type_result["confidence"],
    }
