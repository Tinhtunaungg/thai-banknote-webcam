import cv2


MIN_CONTOUR_AREA = 5000


def find_banknote_contours(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for contour in contours:
        area = cv2.contourArea(contour)

        if area < MIN_CONTOUR_AREA:
            continue

        x, y, width, height = cv2.boundingRect(contour)

        if width == 0 or height == 0:
            continue

        aspect_ratio = width / float(height)

        candidates.append(
            {
                "contour": contour,
                "area": area,
                "bbox": (x, y, width, height),
                "aspect_ratio": aspect_ratio,
            }
        )

    candidates.sort(key=lambda item: item["area"], reverse=True)
    return candidates


def draw_contour_candidates(frame, candidates):
    output = frame.copy()

    if candidates:
        best = candidates[0]
        x, y, width, height = best["bbox"]

        cv2.rectangle(output, (x, y), (x + width, y + height), (0, 255, 0), 3)
        cv2.drawContours(output, [best["contour"]], -1, (0, 180, 255), 2)

        label = f"candidate area={int(best['area'])} ratio={best['aspect_ratio']:.2f}"
        cv2.putText(
            output,
            label,
            (x, max(25, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    cv2.rectangle(output, (10, 10), (410, 58), (0, 0, 0), -1)
    cv2.putText(
        output,
        f"Contour candidates: {len(candidates)}",
        (22, 43),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.72,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return output
