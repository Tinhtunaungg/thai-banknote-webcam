import argparse
import base64

import cv2
import numpy as np
from flask import Flask, jsonify, render_template, request

from .classifier import (
    SAMPLES_PATH,
    classify_signature,
    compute_signature,
    estimate_white_balance_gains,
    load_samples,
    value_for_label,
)
from .contours import find_banknote_contours
from .perspective import crop_banknote
from .preprocessing import preprocess_frame
from .autocalibration import choose_settings, load_settings, save_settings
from .foreground import refine_banknote
from .segmentation import DEFAULT_SETTINGS, segment_banknote
from .template_matcher import (
    accepted_label,
    decide,
    load_templates,
    match_in_frame,
    match_templates,
    strong_label,
    templates_signature,
)


app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

# Reference data is reloaded automatically when files change on disk,
# so new templates/samples saved by main.py are used without a restart.
_cache = {"samples_mtime": None, "samples": {}, "tpl_sig": None, "templates": {}}


def current_references():
    samples_mtime = SAMPLES_PATH.stat().st_mtime if SAMPLES_PATH.exists() else None
    if samples_mtime != _cache["samples_mtime"]:
        _cache["samples"] = load_samples()
        _cache["samples_mtime"] = samples_mtime

    signature = templates_signature()
    if signature != _cache["tpl_sig"]:
        _cache["templates"] = load_templates()
        _cache["tpl_sig"] = signature

    return _cache["samples"], _cache["templates"]


@app.route("/")
def index():
    return render_template("index.html", calibration_settings=load_settings())


def settings_from_form(form):
    settings = load_settings()
    for key, form_key in [
        ("s_min", "s_min"),
        ("v_min", "v_min"),
        ("v_max", "v_max"),
        ("open_iterations", "open"),
        ("close_iterations", "close"),
    ]:
        if form_key in form:
            settings[key] = int(form[form_key])
    return settings


@app.route("/auto_calibrate", methods=["POST"])
def auto_calibrate():
    uploads = request.files.getlist("frames")
    if len(uploads) != 8:
        return jsonify({"error": "Calibration needs eight camera frames. Please retry."}), 400
    frames = []
    for upload in uploads:
        contents = np.frombuffer(upload.read(), np.uint8)
        image = cv2.imdecode(contents, cv2.IMREAD_COLOR) if contents.size else None
        if image is None:
            return jsonify({"error": "Could not read a camera frame. Please retry."}), 400
        frames.append(preprocess_frame(image)["original"])
    settings, message = choose_settings(frames)
    if settings is None:
        message = message.replace("press A", "tap Auto Calibrate")
        return jsonify({"error": message, "kept_previous_settings": True}), 422
    try:
        save_settings(settings)
    except OSError:
        app.logger.exception("Could not save phone calibration")
        return jsonify({"error": "Could not save calibration. Previous settings kept."}), 503
    processed = preprocess_frame(frames[-1])
    mask = segment_banknote(processed, settings)["mask"]
    encoded, buffer = cv2.imencode(".png", mask)
    result = {"settings": settings, "message": "Calibration saved. Detection resumed."}
    if encoded:
        result["mask"] = "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")
    return jsonify(result)


@app.route("/process_frame", methods=["POST"])
def process_frame():
    upload = request.files.get("frame")
    if upload is None:
        return jsonify({"error": "no frame uploaded"}), 400

    file_bytes = np.frombuffer(upload.read(), dtype=np.uint8)
    frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({"error": "could not decode frame"}), 400

    settings = settings_from_form(request.form)

    processed = preprocess_frame(frame)
    segmented = segment_banknote(processed, settings)
    candidates = find_banknote_contours(segmented["mask"])
    best_candidate = candidates[0] if candidates else None
    cropped_banknote, _ = (
        crop_banknote(processed["original"], best_candidate) if best_candidate else (None, None)
    )

    foreground = refine_banknote(
        cropped_banknote, enabled=request.form.get("use_grabcut", "1") != "0"
    )
    wb_gains = estimate_white_balance_gains(processed["original"])
    signature = (
        compute_signature(cropped_banknote, wb_gains, mask=foreground["mask"])
        if foreground["usable"] else None
    )

    samples, templates = current_references()
    color_label, color_score = classify_signature(signature, samples)

    if cropped_banknote is not None:
        # Outline found: shape + color must agree
        match = match_templates(cropped_banknote, templates)
        label, reason = decide(accepted_label(match), color_label, bool(templates), bool(samples))
        if reason == "color unsure":
            label = strong_label(match)
            reason = "shape only (color unsure)" if label else "color unsure, weak shape"

        # The outline may be WRONG (e.g. note merged with a white table), so
        # the crop holds no number. Unless color clearly said another
        # denomination, also search the whole frame.
        if label is None and reason != "shape/color disagree - try again":
            frame_match = match_in_frame(processed["original"], templates)
            frame_label = accepted_label(frame_match)
            if frame_label is not None:
                match, label = frame_match, frame_label
                reason = "shape only (whole frame)"
    else:
        # No outline (hand / busy background): search numbers in the whole frame
        match = match_in_frame(processed["original"], templates)
        label = accepted_label(match)
        reason = "shape only (no outline found)" if label else "no banknote found"

    result = {
        "label": label,
        "value": value_for_label(label) if label else 0,
        "score": float(color_score) if color_score is not None else None,
        "reason": reason,
        "foreground_status": foreground["status"],
        "foreground_coverage": foreground["coverage"],
        "color_label": color_label,
        "template_label": match["label"] if match else None,
        "template_score": match["score"] if match else None,
    }

    # Debug line in the PC terminal for every frame
    print(
        f"shape={result['template_label']} "
        f"{result['template_score'] or 0:.2f} | "
        f"color={result['color_label']} | "
        f"final={result['label']} ({result['reason']})"
    )

    if request.form.get("include_mask") == "1":
        encoded, buffer = cv2.imencode(".jpg", segmented["mask"])
        if encoded:
            result["mask"] = "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")

    if request.form.get("include_mask") == "1" and foreground["mask"] is not None:
        encoded, buffer = cv2.imencode(".png", foreground["mask"])
        if encoded:
            result["foreground_mask"] = "data:image/png;base64," + base64.b64encode(buffer).decode("ascii")

    return jsonify(result)


def parse_args():
    parser = argparse.ArgumentParser(description="Web app version of the banknote detector.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument(
        "--https",
        action="store_true",
        help="Serve over HTTPS with a self-signed cert. Required for camera access from a phone.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    ssl_context = "adhoc" if args.https else None
    app.run(host="0.0.0.0", port=args.port, threaded=True, ssl_context=ssl_context)


if __name__ == "__main__":
    main()
