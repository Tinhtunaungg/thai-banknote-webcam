"""Guided threshold calibration; never changes denomination colour samples."""

import itertools
import json
import math

import cv2
import numpy as np

from .paths import ASSETS_DIR, PROJECT_ROOT
from .preprocessing import camera_warning
from .segmentation import DEFAULT_SETTINGS, segment_banknote


SETTINGS_PATH = PROJECT_ROOT / "calibration_settings.json"
PACKAGED_SETTINGS_PATH = ASSETS_DIR / "calibration_defaults.json"


def guide_box(shape):
    h, w = shape[:2]
    return (int(w * .225), int(h * .275), int(w * .775), int(h * .725))


def validate_settings(settings):
    if not isinstance(settings, dict):
        raise ValueError("Calibration settings must be an object")
    clean = {}
    for key in DEFAULT_SETTINGS:
        value = settings.get(key)
        limit = 5 if key.endswith("iterations") else 255
        if type(value) is not int or not 0 <= value <= limit:
            raise ValueError(f"Invalid calibration setting: {key}")
        clean[key] = value
    if clean["v_min"] > clean["v_max"]:
        raise ValueError("Brightness minimum exceeds maximum")
    return clean


def load_settings(path=SETTINGS_PATH):
    try:
        return validate_settings(json.loads(path.read_text()))
    except FileNotFoundError:
        if path == SETTINGS_PATH:
            return load_settings(PACKAGED_SETTINGS_PATH)
    except (OSError, ValueError, TypeError):
        pass
    return dict(DEFAULT_SETTINGS)


def save_settings(settings, path=SETTINGS_PATH):
    clean = validate_settings(settings)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(clean, indent=2) + "\n")
    temporary.replace(path)


def mask_quality(mask):
    """Score a note-shaped region near the guide, without requiring its size.

    Still score the largest contour, because that is what detection crops.
    """
    h, w = mask.shape
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    contour = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(contour)
    fraction = area / mask.size
    (cx, cy), (rw, rh), _ = cv2.minAreaRect(contour)
    if min(rw, rh) <= 0 or not 1.4 <= max(rw, rh) / min(rw, rh) <= 3.8:
        return 0.0
    if not .012 <= fraction <= .70:
        return 0.0
    rectangularity = min(1.0, area / max(rw * rh, 1))
    if rectangularity < .55:
        return 0.0
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, [contour], -1, 255, -1)
    x1, y1, x2, y2 = guide_box(mask.shape)
    # The note can be smaller than, or extend beyond, the visual guide.
    overlap = np.count_nonzero(filled[y1:y2, x1:x2]) / max(np.count_nonzero(filled), 1)
    if overlap < .25 or not .15 <= cx / w <= .85 or not .12 <= cy / h <= .88:
        return 0.0
    coverage = np.count_nonzero((mask != 0) & (filled != 0)) / max(np.count_nonzero(filled), 1)
    if coverage < .35:
        return 0.0
    centrality = max(0.0, 1.0 - np.hypot(cx / w - .5, cy / h - .5))
    leakage = np.count_nonzero((mask != 0) & (filled == 0)) / mask.size
    return float(.50 * rectangularity + .25 * centrality + .25 * coverage - leakage)


def choose_settings(frames):
    """Search on small frames, then verify the winners at processing size.

    Frames must contain a reasonably flat note near the centre of the guide. A
    coloured rectangle can also satisfy this test: this is calibration, not
    denomination recognition or proof that the object is a banknote.
    """
    if len(frames) < 3:
        return None, "Hold the note steady for more frames"
    if sum(bool(camera_warning(frame)) for frame in frames) > len(frames) // 2:
        return None, "Camera image too dark; restore the live picture, then press A"
    selected = [frames[0], frames[len(frames)//2], frames[-1]]
    if any(f.shape != selected[0].shape for f in selected):
        return None, "Camera size changed; press A to retry"
    small = [cv2.resize(f, (round(f.shape[1] * 480 / max(f.shape[:2])),
                                round(f.shape[0] * 480 / max(f.shape[:2]))))
             for f in selected]
    x1, y1, x2, y2 = guide_box(small[0].shape)
    hsvs = [cv2.cvtColor(f, cv2.COLOR_BGR2HSV) for f in small]
    pixels = np.concatenate([h[y1:y2, x1:x2].reshape(-1, 3) for h in hsvs])
    # Scene-wide Otsu thresholds give both bright-note/dark-background and
    # dark-note/bright-background options. Guide percentiles alone previously
    # missed those splits when background occupied much of the guide.
    saturation = np.concatenate([h[:, :, 1].reshape(-1) for h in hsvs]).astype(np.uint8)
    brightness = np.concatenate([h[:, :, 2].reshape(-1) for h in hsvs]).astype(np.uint8)
    s_split = int(cv2.threshold(saturation, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    v_split = int(cv2.threshold(brightness, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[0])
    sats = sorted({0, 10, 20, 40, 80, min(255, s_split + 1),
                   max(0, int(np.percentile(pixels[:, 1], 25)) - 10)})
    lows = sorted({0, min(255, v_split + 1),
                   max(0, int(np.percentile(pixels[:, 2], 10)) - 15)})
    highs = sorted({255, v_split,
                    min(255, int(np.percentile(pixels[:, 2], 90)) + 8)})

    def stable_score(scores):
        # Require a useful mask in two frames; tolerate one poor exposure frame.
        valid = [score for score in scores if score >= .65]
        return float(np.median(valid)) if len(valid) >= 2 else 0.0

    ranked = []
    for s, low, high, opening, closing in itertools.product(sats, lows, highs,
                                                           (0, 1, 2), (0, 1, 2, 3, 5)):
        settings = dict(s_min=s, v_min=low, v_max=high,
                        open_iterations=opening, close_iterations=closing)
        if low >= high:
            continue
        scores = [mask_quality(segment_banknote({"original": f, "hsv": hsv}, settings)["mask"])
                  for f, hsv in zip(small, hsvs)]
        score = stable_score(scores)
        if score >= .65:
            breadth = ((high - low) / 255 + (255 - s) / 255) / 2
            ranked.append((score + .008 * breadth - .0005 * (opening + closing), settings))
    ranked.sort(key=lambda item: item[0], reverse=True)
    full = [(f, cv2.cvtColor(f, cv2.COLOR_BGR2HSV)) for f in selected]
    best, best_score = None, .65
    for _, settings in ranked[:12]:
        score = stable_score([mask_quality(segment_banknote({"original": f, "hsv": hsv}, settings)["mask"])
                              for f, hsv in full])
        # Among similarly good masks prefer a broader brightness range, so
        # calibration is less tightly fitted to one denomination/exposure.
        breadth = ((settings["v_max"] - settings["v_min"]) / 255
                   + (255 - settings["s_min"]) / 255) / 2
        adjusted = score + .008 * breadth - .0005 * (
            settings["open_iterations"] + settings["close_iterations"])
        if score >= .65 and adjusted > best_score:
            best, best_score = settings, adjusted
    if best is None:
        return None, "Could not separate note from background. Centre one full note, avoid glare, and press A"
    return best, "Auto calibration ready"


class GuidedCalibration:
    def __init__(self):
        self.active = False
        self.message = ""
        self.frames = []

    def start(self, now):
        self.active = True
        self.frames = []
        self.ready_at = now + 3.0
        self.next_sample = self.ready_at
        self.message = "Align note inside guide; C cancels"

    def cancel(self):
        self.active = False
        self.frames = []
        self.message = "Calibration cancelled; previous settings kept"

    def update(self, frame, now):
        if not self.active:
            return None
        if now < self.ready_at:
            self.message = f"Align note inside guide - starting in {math.ceil(self.ready_at-now)}s"
            return None
        if now < self.next_sample:
            return None
        self.frames.append(frame.copy())
        self.next_sample = now + .25
        self.message = f"Hold steady - collecting {len(self.frames)}/8"
        if len(self.frames) < 8:
            return None
        self.active = False
        settings, self.message = choose_settings(self.frames)
        self.frames = []
        return settings


def draw_guide(frame, message):
    output = frame.copy()
    x1, y1, x2, y2 = guide_box(output.shape)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)
    cv2.rectangle(output, (0, 0), (output.shape[1], 80), (0, 0, 0), -1)
    cv2.putText(output, message, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output, "Centre the full note near the guide; exact fit is not needed. C = cancel",
                (15, 60), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 1, cv2.LINE_AA)
    return output
