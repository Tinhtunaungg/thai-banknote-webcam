import cv2
import numpy as np


DEFAULT_SETTINGS = {
    "s_min": 0,
    "v_min": 247,
    "v_max": 255,
    "open_iterations": 0,
    "close_iterations": 5,
}


def segment_banknote(processed, settings=None):
    settings = settings or DEFAULT_SETTINGS
    original = processed["original"]
    hsv = processed["hsv"]

    hue, saturation, value = cv2.split(hsv)

    # Banknotes are usually more colorful than a plain table/background.
    # Keep pixels that are not too gray, not too dark, and not overexposed.
    saturation_mask = cv2.inRange(saturation, settings["s_min"], 255)
    value_mask = cv2.inRange(value, settings["v_min"], settings["v_max"])
    mask = cv2.bitwise_and(saturation_mask, value_mask)

    kernel_small = np.ones((5, 5), np.uint8)
    kernel_large = np.ones((11, 11), np.uint8)

    if settings["open_iterations"] > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_OPEN,
            kernel_small,
            iterations=settings["open_iterations"],
        )

    if settings["close_iterations"] > 0:
        mask = cv2.morphologyEx(
            mask,
            cv2.MORPH_CLOSE,
            kernel_large,
            iterations=settings["close_iterations"],
        )

    segmented = cv2.bitwise_and(original, original, mask=mask)

    return {
        "saturation": saturation,
        "value": value,
        "mask": mask,
        "segmented": segmented,
        "settings": settings,
    }


def draw_segmentation_status(frame):
    output = frame.copy()
    cv2.rectangle(output, (10, 10), (520, 58), (0, 0, 0), -1)
    cv2.putText(
        output,
        "Segmented banknote area: white mask = kept pixels",
        (22, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output
