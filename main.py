import argparse
import sys
import time

import cv2
import numpy as np

from classifier import (
    LEARN_KEYS,
    add_sample,
    classify_signature,
    compute_signature,
    draw_classification,
    estimate_white_balance_gains,
    load_samples,
    value_for_label,
)
from contours import draw_contour_candidates, find_banknote_contours
from perspective import crop_banknote, draw_perspective_rectangle
from preprocessing import camera_warning, draw_status, preprocess_frame
from autocalibration import GuidedCalibration, draw_guide, load_settings, save_settings
from foreground import refine_banknote
from segmentation import DEFAULT_SETTINGS, segment_banknote
from template_matcher import (
    accepted_label,
    decide,
    draw_template_match,
    load_templates,
    match_templates,
    save_template,
)


COMBINED_WINDOW = "Thai Banknote Detector"
ROI_WINDOW = "Select the number, then press ENTER (c = cancel)"
PANEL_WIDTH = 480
PANEL_HEIGHT = 270


def open_capture(index):
    # macOS needs AVFoundation; Windows works best with DirectShow; Linux uses the default.
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if sys.platform.startswith("win"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def nothing(_):
    pass


def parse_args():
    parser = argparse.ArgumentParser(description="Thai banknote detector (OpenCV).")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show every intermediate image-processing window.",
    )
    parser.add_argument("--auto-calibrate", action="store_true",
                        help="Start automatic calibration from live camera frames.")
    parser.add_argument("--no-grabcut", action="store_true",
                        help="Disable foreground refinement for comparison.")
    return parser.parse_args()


TRACKBARS = {
    "s_min": "S min", "v_min": "V min", "v_max": "V max",
    "open_iterations": "Open", "close_iterations": "Close",
}


def create_calibration_trackbars(settings):
    cv2.namedWindow(COMBINED_WINDOW)
    for key, title in TRACKBARS.items():
        maximum = 5 if key.endswith("iterations") else 255
        cv2.createTrackbar(title, COMBINED_WINDOW, settings[key], maximum, nothing)


def apply_calibration_settings(settings):
    for key, title in TRACKBARS.items():
        cv2.setTrackbarPos(title, COMBINED_WINDOW, settings[key])


def read_calibration_settings():
    settings = {
        "s_min": cv2.getTrackbarPos("S min", COMBINED_WINDOW),
        "v_min": cv2.getTrackbarPos("V min", COMBINED_WINDOW),
        "v_max": cv2.getTrackbarPos("V max", COMBINED_WINDOW),
        "open_iterations": cv2.getTrackbarPos("Open", COMBINED_WINDOW),
        "close_iterations": cv2.getTrackbarPos("Close", COMBINED_WINDOW),
    }
    if settings["v_max"] < settings["v_min"]:
        settings["v_max"] = settings["v_min"]
    return settings


def put_text(image, text, origin, size, color):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, size, color, 2, cv2.LINE_AA)


def draw_segmentation_mask(mask, settings):
    output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    text = (
        f"S {settings['s_min']} | V {settings['v_min']}-{settings['v_max']} | "
        f"Open {settings['open_iterations']} | Close {settings['close_iterations']}"
    )
    cv2.rectangle(output, (10, 64), (520, 106), (0, 0, 0), -1)
    put_text(output, text, (22, 93), 0.62, (255, 255, 255))
    return output


def draw_learning_status(frame, message):
    output = frame.copy()
    cv2.rectangle(output, (10, 112), (860, 154), (0, 0, 0), -1)
    put_text(output, message, (22, 141), 0.62, (255, 255, 255))
    return output


def draw_result(frame, final_label, reason, match, color_label, color_score):
    output = frame.copy()

    if final_label is None:
        line_1 = "Detected: unknown"
        line_2 = "Total: 0 baht"
    else:
        line_1 = f"Detected: {final_label}"
        line_2 = f"Total: {value_for_label(final_label)} baht"

    shape_text = f"{match['label']} {match['score']:.2f}" if match is not None else "none"
    color_text = (
        f"{color_label} {color_score:.2f}"
        if color_label is not None and color_score is not None
        else "none"
    )

    cv2.rectangle(output, (10, 160), (700, 300), (0, 0, 0), -1)
    put_text(output, line_1, (22, 190), 0.72, (0, 255, 255))
    put_text(output, line_2, (22, 220), 0.72, (0, 255, 255))
    put_text(output, f"Shape: {shape_text} | Color: {color_text}", (22, 250), 0.55, (255, 255, 255))
    put_text(output, reason, (22, 280), 0.55, (255, 255, 255))
    return output


def make_panel(image, label):
    if image is None:
        panel = np.zeros((PANEL_HEIGHT, PANEL_WIDTH, 3), dtype=np.uint8)
    else:
        panel = cv2.resize(image, (PANEL_WIDTH, PANEL_HEIGHT))
    cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 26), (0, 0, 0), -1)
    cv2.putText(panel, label, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return panel


def instruction_lines(calibrating, message):
    if calibrating:
        instructions = [
            "AUTO CALIBRATION: finding S min, V min, V max, Open and Close",
            "Centre the full note near the guide. Exact fit is not needed; keep fingers outside.",
            "C = cancel and keep previous settings | A = restart | Q = quit",
        ]
    else:
        instructions = [
            "A = auto-calibrate | S = save sliders | T = number-template mode | Q = quit",
            "Auto calibration samples the live feed, then saves and locks the settings.",
            "Check mask: note white, background black. Adjust sliders or press A to retry.",
            "Save colour samples: 2 = 20 | 5 = 50 | 1 = 100 | b = 500 | 0 = 1000 baht",
        ]
    return instructions + ["Status: " + message, "Click this window before pressing a key."]


def draw_instructions(width, calibrating, message):
    # A separate panel keeps instructions off the banknote and guide.
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = .52
    lines = []
    for text in instruction_lines(calibrating, message):
        line = ""
        for word in text.split():
            proposed = (line + " " + word).strip()
            if line and cv2.getTextSize(proposed, font, scale, 1)[0][0] > width - 32:
                lines.append(line)
                line = word
            else:
                line = proposed
        lines.append(line)
    panel = np.full((20 + 25 * len(lines), width, 3), (28, 28, 28), np.uint8)
    for index, line in enumerate(lines):
        colour = (0, 255, 255) if index == 0 or line.startswith("Status:") else (230, 230, 230)
        cv2.putText(panel, line, (16, 25 + index * 25), font, scale, colour, 1, cv2.LINE_AA)
    return panel


def build_combined_view(camera_output, mask_bgr, classified_crop,
                        calibrating=False, message="Ready"):
    mask_panel = make_panel(mask_bgr, "2 - Segmentation Mask")
    crop_panel = make_panel(classified_crop, "3 - Crop after GrabCut")
    bottom_row = cv2.hconcat([mask_panel, crop_panel])
    if bottom_row.shape[1] != camera_output.shape[1]:
        bottom_row = cv2.resize(bottom_row, (camera_output.shape[1], PANEL_HEIGHT))
    instructions = draw_instructions(camera_output.shape[1], calibrating, message)
    return cv2.vconcat([camera_output, bottom_row, instructions])


def main():
    args = parse_args()
    cap = open_capture(args.camera)
    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open camera {args.camera}. Run camera_scan.py to find the right index."
        )
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    create_calibration_trackbars(load_settings())
    calibration = GuidedCalibration()
    if args.auto_calibrate:
        calibration.start(time.monotonic())
    samples = load_samples()
    templates = load_templates()
    template_mode = False
    last_message = "A=auto calibrate | S=save sliders | T then denomination key=number template"
    previous_time = time.time()

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        settings = read_calibration_settings()

        # Segmentation -> contour -> perspective crop
        processed = preprocess_frame(frame)
        warning = camera_warning(frame)
        if warning and calibration.active:
            calibration.cancel()
            last_message = "Restore the live camera picture, then press A to calibrate"
        was_calibrating = calibration.active
        automatic = calibration.update(processed["original"], time.monotonic())
        if automatic is not None:
            apply_calibration_settings(automatic)
            settings = automatic
            try:
                save_settings(automatic)
                calibration.message = "Auto settings saved. A=retry | S=save manual settings"
            except OSError as error:
                calibration.message = "Settings applied, but could not save; see terminal"
                print(f"Could not save calibration: {error}")
        if was_calibrating and not calibration.active:
            last_message = calibration.message
            print(last_message)
        segmented = segment_banknote(processed, settings)
        candidates = find_banknote_contours(segmented["mask"])
        best_candidate = candidates[0] if candidates else None
        cropped_banknote, _ = (
            crop_banknote(processed["original"], best_candidate) if best_candidate else (None, None)
        )

        foreground = refine_banknote(
            None if calibration.active else cropped_banknote, enabled=not args.no_grabcut
        )

        # Check 1: color (histogram similarity)
        wb_gains = estimate_white_balance_gains(processed["original"])
        current_signature = (
            compute_signature(cropped_banknote, wb_gains, mask=foreground["mask"])
            if foreground["usable"] else None
        )
        color_label, color_score = classify_signature(current_signature, samples)

        # Check 2: shape of the printed number (template matching, NCC)
        match = match_templates(None if calibration.active else cropped_banknote, templates)

        # Final answer: both checks must agree
        final_label, reason = decide(accepted_label(match), color_label, bool(templates), bool(samples))
        if template_mode:
            reason = "TEMPLATE MODE: press 2/5/1/b/0, then drag a box around the number"

        now = time.time()
        fps = 1.0 / max(now - previous_time, 0.001)
        previous_time = now

        camera_output = draw_perspective_rectangle(processed["original"], best_candidate)
        camera_output = draw_status(camera_output, fps)
        if calibration.active:
            camera_output = draw_guide(processed["original"], calibration.message)
        else:
            camera_output = draw_result(camera_output, final_label, reason, match, color_label, color_score)
        if warning:
            camera_output = draw_learning_status(camera_output, "CAMERA IMAGE TOO DARK - check iPhone / lens / lighting")

        classified_crop = None
        if foreground["preview"] is not None:
            sample_counts = {label: len(items) for label, items in samples.items()}
            classified_crop = draw_classification(
                foreground["preview"], current_signature, color_label, color_score, sample_counts
            )
            classified_crop = draw_template_match(classified_crop, match)

        mask_bgr = draw_segmentation_mask(segmented["mask"], settings)
        cv2.imshow(COMBINED_WINDOW, build_combined_view(
            camera_output, mask_bgr, classified_crop, calibration.active,
            warning or (calibration.message if calibration.active else last_message),
        ))

        if args.debug:
            if foreground["mask"] is not None:
                cv2.imshow("Debug - GrabCut Mask", foreground["mask"])
                cv2.imshow("Debug - GrabCut Foreground", foreground["preview"])
            cv2.imshow("Debug - Gray", processed["gray"])
            cv2.imshow("Debug - Blurred", processed["blurred"])
            cv2.imshow("Debug - Segmented Banknote", segmented["segmented"])
            cv2.imshow("Debug - Contour Candidate", draw_contour_candidates(processed["original"], candidates))

        key = cv2.waitKey(1) & 0xFF

        if key == ord("a"):
            calibration.start(time.monotonic())
        elif key == ord("c") and calibration.active:
            calibration.cancel()
            last_message = calibration.message
        elif key == ord("s") and not calibration.active:
            try:
                save_settings(read_calibration_settings())
                last_message = "Manual settings saved"
            except OSError as error:
                last_message = "Could not save settings; see terminal"
                print(f"Could not save calibration: {error}")
        elif key == ord("t") and not calibration.active:
            template_mode = not template_mode
            last_message = "template mode ON" if template_mode else "template mode OFF"
            print(last_message)

        elif key in LEARN_KEYS and template_mode and not calibration.active:
            label = LEARN_KEYS[key]
            template_mode = False
            if cropped_banknote is None:
                last_message = "no banknote crop - cannot save template"
            else:
                frozen = cropped_banknote.copy()
                roi = cv2.selectROI(ROI_WINDOW, frozen, showCrosshair=True)
                cv2.destroyWindow(ROI_WINDOW)
                path = save_template(frozen, roi, label)
                if path is None:
                    last_message = "template cancelled"
                else:
                    templates = load_templates()
                    last_message = f"saved template {path.name} for {label}"
            print(last_message)

        elif key in LEARN_KEYS and not calibration.active:
            label = LEARN_KEYS[key]
            if add_sample(samples, label, current_signature):
                last_message = f"saved signature as {label}"
            else:
                last_message = "no valid crop/signature to save"
            print(last_message)

        if key == ord("q") or key == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()