"""Template matching (Chapter 11) for Thai banknote denominations.

Idea: keep one small reference image (template) of the printed number for each
denomination, slide it over the flattened banknote crop, and score every
position with Normalized Cross-Correlation (cv2.TM_CCOEFF_NORMED).
The denomination whose template gets the highest score wins.

No model training: templates are cut from real notes once, by hand.
"""

import time

import cv2
import numpy as np

from .paths import reference_path


TEMPLATE_DIR = reference_path("note_templates")

# TM_CCOEFF_NORMED: 1.0 = identical, 0 = unrelated. Below this = "not found".
MATCH_THRESHOLD = 0.60

# Matching runs on a half-size image so it stays real-time.
WORK_SCALE = 0.5

# Small image pyramid: tolerates the note being slightly closer / farther.
SCALES = (0.85, 1.0, 1.15)

LABEL_TO_SLUG = {
    "20 baht": "20",
    "50 baht": "50",
    "100 baht": "100",
    "500 baht": "500",
    "1000 baht": "1000",
}
SLUG_TO_LABEL = {slug: label for label, slug in LABEL_TO_SLUG.items()}


def prepare_gray(image):
    """Grayscale + Gaussian blur (noise removal).

    Matching uses shape, not color, so faded or tinted notes still match.
    No histogram equalization here on purpose: TM_CCOEFF_NORMED already
    normalizes brightness and contrast, and equalizing a flat paper area
    would mostly amplify camera noise.
    """
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def load_templates(directory=TEMPLATE_DIR):
    templates = {}
    if not directory.exists():
        return templates

    for path in sorted(directory.glob("*.png")):
        slug = path.stem.split("_")[0]
        label = SLUG_TO_LABEL.get(slug)
        if label is None:
            continue
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is not None:
            templates.setdefault(label, []).append(image)

    return templates


def templates_signature(directory=TEMPLATE_DIR):
    """Changes whenever a template file is added/removed (used to auto-reload)."""
    if not directory.exists():
        return ()
    return tuple(
        (path.name, path.stat().st_mtime) for path in sorted(directory.glob("*.png"))
    )


def save_template(crop, roi, label, directory=TEMPLATE_DIR):
    """Cut the selected box out of the (preprocessed) crop and save it."""
    x, y, width, height = [int(value) for value in roi]
    if width < 10 or height < 10:
        return None

    directory.mkdir(parents=True, exist_ok=True)
    gray = prepare_gray(crop)
    template = gray[y : y + height, x : x + width]
    path = directory / f"{LABEL_TO_SLUG[label]}_{int(time.time() * 1000)}.png"
    cv2.imwrite(str(path), template)
    return path


def match_templates(crop, templates, scales=SCALES):
    """Return the best match over all templates, scales and both orientations."""
    if crop is None or not templates:
        return None

    gray = prepare_gray(crop)
    small = cv2.resize(gray, None, fx=WORK_SCALE, fy=WORK_SCALE)
    small_h, small_w = small.shape[:2]

    # The note may be held upside down: also search the 180-degree rotation.
    orientations = [(0, small), (180, cv2.rotate(small, cv2.ROTATE_180))]

    scores = {}
    best = None

    for label, label_templates in templates.items():
        for template in label_templates:
            for scale in scales:
                tw = int(template.shape[1] * WORK_SCALE * scale)
                th = int(template.shape[0] * WORK_SCALE * scale)
                if tw < 8 or th < 8 or tw > small_w or th > small_h:
                    continue
                resized = cv2.resize(template, (tw, th))

                for angle, image in orientations:
                    result = cv2.matchTemplate(image, resized, cv2.TM_CCOEFF_NORMED)
                    _, max_score, _, max_loc = cv2.minMaxLoc(result)

                    if max_score > scores.get(label, -1.0):
                        scores[label] = max_score

                    if best is None or max_score > best["score"]:
                        x, y = max_loc
                        if angle == 180:
                            # Map the box back to the un-rotated crop.
                            x = small_w - x - tw
                            y = small_h - y - th
                        best = {
                            "label": label,
                            "score": float(max_score),
                            "box": (
                                int(x / WORK_SCALE),
                                int(y / WORK_SCALE),
                                int(tw / WORK_SCALE),
                                int(th / WORK_SCALE),
                            ),
                            "angle": angle,
                        }

    if best is None:
        return None

    # "50" is inside "500" and "100" is inside "1000" (same for Thai digits).
    # If the longer number also matches almost as well, the shorter template
    # was only matching part of it -> choose the longer number.
    best_slug = LABEL_TO_SLUG[best["label"]]
    for label, score in scores.items():
        slug = LABEL_TO_SLUG[label]
        if len(slug) > len(best_slug) and slug.startswith(best_slug):
            if score >= best["score"] - CONTAINED_MARGIN:
                best["label"] = label
                best["score"] = float(score)

    best["scores"] = {label: float(score) for label, score in scores.items()}
    best["accepted"] = best["score"] >= MATCH_THRESHOLD
    return best


# How close the longer number's score must be to win over a contained one.
CONTAINED_MARGIN = 0.10

# Fallback when no banknote outline is found (busy background, hand, glare):
# search the printed numbers in the WHOLE frame. The note size in the frame is
# unknown, so a wider image pyramid is used, and a stricter threshold.
FRAME_SCALES = (0.5, 0.57, 0.65, 0.75, 0.86, 1.0, 1.15, 1.3)
FRAME_THRESHOLD = 0.70


# Whole-frame mode has no color check, so the winner must also beat the
# runner-up denomination clearly (otherwise answer "not sure").
FRAME_MARGIN = 0.05


def match_in_frame(frame, templates):
    match = match_templates(frame, templates, scales=FRAME_SCALES)
    if match is not None:
        others = [s for label, s in match["scores"].items() if label != match["label"]]
        runner_up = max(others) if others else 0.0
        contained = LABEL_TO_SLUG[match["label"]]
        # a longer number that contains the runner-up (500 vs 50) is fine
        clear = (match["score"] - runner_up >= FRAME_MARGIN) or any(
            contained.startswith(LABEL_TO_SLUG[label]) and label != match["label"]
            and s == runner_up
            for label, s in match["scores"].items()
        )
        match["accepted"] = match["score"] >= FRAME_THRESHOLD and clear
    return match


def strong_label(match):
    """Shape-only answer: needs the stricter FRAME_THRESHOLD."""
    if match is None or match["score"] < FRAME_THRESHOLD:
        return None
    return match["label"]


def accepted_label(match):
    if match is None or not match["accepted"]:
        return None
    return match["label"]


def decide(template_label, color_label, have_templates, have_color):
    """Final answer from the two independent checks.

    shape (template matching) + color (histogram similarity) must agree.
    This blocks mistakes such as reading the "100" inside "1000".
    """
    if have_templates and have_color:
        if template_label is None:
            return None, "no number found"
        if color_label == template_label:
            return template_label, "shape + color agree"
        if color_label is None:
            # Color unsure (other camera / lighting): trust the shape only
            # when it is a strong match (see strong_label()).
            return None, "color unsure"
        return None, "shape/color disagree - try again"

    if have_templates:
        return template_label, "shape only"
    if have_color:
        return color_label, "color only"
    return None, "no templates or color samples yet"


def draw_template_match(image, match):
    output = image.copy()
    if match is None:
        return output

    x, y, width, height = match["box"]
    color = (0, 255, 0) if match["accepted"] else (0, 0, 255)
    cv2.rectangle(output, (x, y), (x + width, y + height), color, 3)
    cv2.putText(
        output,
        f"{match['label']} NCC={match['score']:.2f}",
        (x, max(22, y - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        color,
        2,
        cv2.LINE_AA,
    )
    return output
