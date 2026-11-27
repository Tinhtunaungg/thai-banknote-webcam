import argparse

import cv2

from .classifier import (
    classify_signature,
    compute_signature,
    draw_classification,
    estimate_white_balance_gains,
    load_samples,
)
from .foreground import refine_banknote
from .contours import draw_contour_candidates, find_banknote_contours
from .perspective import crop_banknote, draw_perspective_rectangle
from .preprocessing import preprocess_frame
from .segmentation import DEFAULT_SETTINGS, segment_banknote


def parse_args():
    parser = argparse.ArgumentParser(description="Test preprocessing on one image.")
    parser.add_argument("image", help="Path to the image file.")
    parser.add_argument("--s-min", type=int, default=DEFAULT_SETTINGS["s_min"])
    parser.add_argument("--v-min", type=int, default=DEFAULT_SETTINGS["v_min"])
    parser.add_argument("--v-max", type=int, default=DEFAULT_SETTINGS["v_max"])
    parser.add_argument("--open", type=int, default=DEFAULT_SETTINGS["open_iterations"])
    parser.add_argument("--close", type=int, default=DEFAULT_SETTINGS["close_iterations"])
    parser.add_argument("--no-grabcut", action="store_true",
                        help="Use the original crop without foreground refinement.")
    return parser.parse_args()


def main():
    args = parse_args()
    frame = cv2.imread(args.image)

    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")

    processed = preprocess_frame(frame)
    settings = {
        "s_min": args.s_min,
        "v_min": args.v_min,
        "v_max": args.v_max,
        "open_iterations": args.open,
        "close_iterations": args.close,
    }
    segmented = segment_banknote(processed, settings)
    candidates = find_banknote_contours(segmented["mask"])
    best_candidate = candidates[0] if candidates else None
    contour_output = draw_contour_candidates(processed["original"], candidates)
    rectangle_output = draw_perspective_rectangle(processed["original"], best_candidate)
    cropped_banknote, _ = (
        crop_banknote(processed["original"], best_candidate)
        if best_candidate
        else (None, None)
    )
    samples = load_samples()
    foreground = refine_banknote(cropped_banknote, enabled=not args.no_grabcut)
    wb_gains = estimate_white_balance_gains(processed["original"])
    signature = (
        compute_signature(cropped_banknote, wb_gains, mask=foreground["mask"])
        if foreground["usable"] else None
    )
    predicted_label, score = classify_signature(signature, samples)

    cv2.imwrite("preprocess_original.jpg", processed["original"])
    cv2.imwrite("preprocess_gray.jpg", processed["gray"])
    cv2.imwrite("preprocess_blurred.jpg", processed["blurred"])
    cv2.imwrite("segment_mask.jpg", segmented["mask"])
    cv2.imwrite("segment_banknote.jpg", segmented["segmented"])
    cv2.imwrite("contour_candidate.jpg", contour_output)
    cv2.imwrite("rotated_rectangle.jpg", rectangle_output)

    if cropped_banknote is not None:
        sample_counts = {label: len(label_samples) for label, label_samples in samples.items()}
        classified_crop = draw_classification(
            foreground["preview"],
            signature,
            predicted_label,
            score,
            sample_counts,
        )
        cv2.imwrite("perspective_crop.jpg", classified_crop)
        cv2.imwrite("grabcut_mask.png", foreground["mask"])
        cv2.imwrite("grabcut_foreground.jpg", foreground["preview"])

    print("saved preprocess_original.jpg")
    print("saved preprocess_gray.jpg")
    print("saved preprocess_blurred.jpg")
    print("saved segment_mask.jpg")
    print("saved segment_banknote.jpg")
    print("saved contour_candidate.jpg")
    print("saved rotated_rectangle.jpg")
    if cropped_banknote is not None:
        print("saved perspective_crop.jpg")
    print(f"contour candidates: {len(candidates)}")
    print(f"foreground: {foreground['status']} ({foreground['coverage']:.0%} retained)")
    print(f"signature captured: {signature is not None}")
    print(f"predicted label: {predicted_label} (score {score})")


if __name__ == "__main__":
    main()
