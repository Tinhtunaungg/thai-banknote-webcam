import cv2
import numpy as np


MAX_SIDE = 960


def camera_warning(frame):
    """Check source pixels, before overlays or segmentation can hide them."""
    small = cv2.resize(frame, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    if float(gray.mean()) < 8 and float(np.percentile(gray, 95)) < 15:
        return "Camera image is too dark: check lens, lighting and iPhone camera connection"
    return None


def preprocess_frame(frame):
    # Keep the aspect ratio: phones send PORTRAIT frames (e.g. 1080x1920).
    # Forcing them into 960x540 would stretch the banknote and its numbers.
    height, width = frame.shape[:2]
    scale = MAX_SIDE / float(max(height, width))
    resized = cv2.resize(frame, (int(width * scale), int(height * scale)))

    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)

    return {
        "original": resized,
        "gray": gray,
        "blurred": blurred,
        "hsv": hsv,
    }


def draw_status(frame, fps):
    output = frame.copy()
    cv2.rectangle(output, (10, 10), (470, 58), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"Webcam capture + preprocessing | FPS {fps:.1f}",
        (22, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.68,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return output