import cv2
import numpy as np


CROP_WIDTH = 640
CROP_HEIGHT = 300
MIN_NOTE_RATIO = 1.6
MAX_NOTE_RATIO = 3.4


def order_points(points):
    points = np.array(points, dtype="float32")
    ordered = np.zeros((4, 2), dtype="float32")

    point_sum = points.sum(axis=1)
    point_diff = np.diff(points, axis=1)

    ordered[0] = points[np.argmin(point_sum)]
    ordered[2] = points[np.argmax(point_sum)]
    ordered[1] = points[np.argmin(point_diff)]
    ordered[3] = points[np.argmax(point_diff)]

    return ordered


def distance(point_a, point_b):
    return float(np.linalg.norm(point_a - point_b))


def rectangle_from_candidate(candidate):
    rect = cv2.minAreaRect(candidate["contour"])
    (_, _), (width, height), _ = rect

    if width <= 0 or height <= 0:
        return None

    long_side = max(width, height)
    short_side = min(width, height)
    ratio = long_side / short_side

    if ratio < MIN_NOTE_RATIO or ratio > MAX_NOTE_RATIO:
        return None

    box = cv2.boxPoints(rect).astype(int)
    return {
        "box": box,
        "ratio": ratio,
        "size": (width, height),
    }


def crop_banknote(frame, candidate):
    rectangle = rectangle_from_candidate(candidate)

    if rectangle is None:
        return None, None

    source = order_points(rectangle["box"])
    top_left, top_right, bottom_right, bottom_left = source

    top_width = distance(top_left, top_right)
    bottom_width = distance(bottom_left, bottom_right)
    left_height = distance(top_left, bottom_left)
    right_height = distance(top_right, bottom_right)

    target_width = int(max(top_width, bottom_width))
    target_height = int(max(left_height, right_height))

    if target_width <= 0 or target_height <= 0:
        return None, rectangle

    if target_height > target_width:
        target_width, target_height = target_height, target_width
        source = np.array([bottom_left, top_left, top_right, bottom_right], dtype="float32")

    target_width = max(target_width, CROP_WIDTH)
    target_height = max(target_height, CROP_HEIGHT)

    destination = np.array(
        [
            [0, 0],
            [target_width - 1, 0],
            [target_width - 1, target_height - 1],
            [0, target_height - 1],
        ],
        dtype="float32",
    )

    transform = cv2.getPerspectiveTransform(source, destination)
    cropped = cv2.warpPerspective(frame, transform, (target_width, target_height))

    if cropped.shape[0] > cropped.shape[1]:
        cropped = cv2.rotate(cropped, cv2.ROTATE_90_CLOCKWISE)

    cropped = cv2.resize(cropped, (CROP_WIDTH, CROP_HEIGHT))
    return cropped, rectangle


def draw_perspective_rectangle(frame, candidate):
    output = frame.copy()

    if candidate is None:
        return output

    rectangle = rectangle_from_candidate(candidate)

    if rectangle is None:
        cv2.putText(
            output,
            "candidate rejected by aspect ratio",
            (22, 43),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )
        return output

    cv2.drawContours(output, [rectangle["box"]], 0, (255, 0, 255), 3)
    cv2.putText(
        output,
        f"rotated rectangle ratio={rectangle['ratio']:.2f} | crop auto-landscape",
        (22, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 0, 255),
        2,
        cv2.LINE_AA,
    )
    return output
