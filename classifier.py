import json
from pathlib import Path

import cv2
import numpy as np


SAMPLES_PATH = Path("hist_samples.json")
LEARN_KEYS = {
    ord("2"): "20 baht",
    ord("5"): "50 baht",
    ord("1"): "100 baht",
    ord("b"): "500 baht",
    ord("0"): "1000 baht",
}
DENOMINATION_VALUES = {
    "20 baht": 20,
    "50 baht": 50,
    "100 baht": 100,
    "500 baht": 500,
    "1000 baht": 1000,
}

HUE_BINS = 30
SATURATION_BINS = 32

# cv2.HISTCMP_CORREL: 1.0 = identical, 0.0 = unrelated. Below this, call it "unknown"
# instead of confidently naming the closest denomination.
MATCH_THRESHOLD = 0.5


def load_samples(path=SAMPLES_PATH):
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    return {
        label: [np.array(histogram, dtype="float32") for histogram in histograms]
        for label, histograms in data.items()
    }


def save_samples(samples, path=SAMPLES_PATH):
    data = {
        label: [histogram.tolist() for histogram in histograms]
        for label, histograms in samples.items()
    }

    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=2)


def center_region(image):
    height, width = image.shape[:2]
    x1 = int(width * 0.25)
    x2 = int(width * 0.75)
    y1 = int(height * 0.25)
    y2 = int(height * 0.75)
    return image[y1:y2, x1:x2]


def estimate_white_balance_gains(full_frame):
    # Gray-world assumption: a whole camera frame (mostly background/table, note is
    # a small part of it) averages out close to gray. Must be estimated from the
    # FULL frame, not the note crop alone — a crop that's almost entirely one
    # strong color has a mean equal to that color, so "correcting it to gray"
    # would erase the note's own color instead of the light source's tint.
    image = full_frame.astype("float32")
    blue, green, red = cv2.split(image)

    gray_average = (blue.mean() + green.mean() + red.mean()) / 3.0

    return (
        gray_average / max(blue.mean(), 1e-6),
        gray_average / max(green.mean(), 1e-6),
        gray_average / max(red.mean(), 1e-6),
    )


def apply_white_balance(image, gains):
    blue_gain, green_gain, red_gain = gains
    balanced = image.astype("float32")
    blue, green, red = cv2.split(balanced)

    blue *= blue_gain
    green *= green_gain
    red *= red_gain

    balanced = cv2.merge([blue, green, red])
    return np.clip(balanced, 0, 255).astype("uint8")


def compute_signature(image, white_balance_gains=None, mask=None):
    if image is None:
        return None

    region = center_region(image)
    region_mask = None
    if mask is not None:
        if mask.shape != image.shape[:2]:
            raise ValueError("Foreground mask must match the image dimensions")
        region_mask = np.where(center_region(mask) != 0, 255, 0).astype("uint8")
        # Avoid making a decision or learning a sample from a nearly hidden note.
        if cv2.countNonZero(region_mask) < max(20, region_mask.size * 0.25):
            return None
    if white_balance_gains is not None:
        region = apply_white_balance(region, white_balance_gains)
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)

    # Hue + Saturation only, no Value: keeps the signature stable when brightness
    # changes (shadow, distance from light) without touching color identity.
    histogram = cv2.calcHist(
        [hsv], [0, 1], region_mask, [HUE_BINS, SATURATION_BINS], [0, 180, 0, 256]
    )
    cv2.normalize(histogram, histogram, 0, 1, cv2.NORM_MINMAX)
    return histogram


def classify_signature(signature, samples):
    if signature is None or not samples:
        return None, None

    best_label = None
    best_score = None

    for label, label_samples in samples.items():
        for reference in label_samples:
            score = cv2.compareHist(signature, reference, cv2.HISTCMP_CORREL)

            if best_score is None or score > best_score:
                best_score = score
                best_label = label

    if best_score is None or best_score < MATCH_THRESHOLD:
        return None, best_score

    return best_label, best_score


def value_for_label(label):
    return DENOMINATION_VALUES.get(label, 0)


def add_sample(samples, label, signature):
    if signature is None:
        return False

    samples.setdefault(label, []).append(signature)
    save_samples(samples)
    return True


def draw_classification(crop, signature, label, score, sample_counts):
    output = crop.copy()

    cv2.rectangle(output, (0, 0), (output.shape[1], 76), (0, 0, 0), -1)

    line_1 = "Signature: none" if signature is None else "Signature: HS histogram, white-balanced"

    score_text = f"{score:.2f}" if score is not None else "n/a"
    if label is None:
        total_samples = sum(sample_counts.values())
        line_2 = f"Class: unknown | best score {score_text} | samples: {total_samples}"
    else:
        line_2 = f"Class: {label} | match score: {score_text}"

    cv2.putText(
        output,
        line_1,
        (14, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        output,
        line_2,
        (14, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    h, w = output.shape[:2]
    x1 = int(w * 0.25)
    x2 = int(w * 0.75)
    y1 = int(h * 0.25)
    y2 = int(h * 0.75)
    cv2.rectangle(output, (x1, y1), (x2, y2), (0, 255, 255), 2)

    return output
