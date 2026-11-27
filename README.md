# Thai Banknote Detector

Recognize a single Thai banknote using a desktop webcam or a phone camera. The detector supports **฿20, ฿50, ฿100, ฿500, and ฿1,000**. It combines OpenCV segmentation, perspective correction, GrabCut foreground refinement, color histograms, and printed-number template matching.

## Setup

Use Python 3.13 and run the setup from the repository root. On macOS and Linux:

```bash
python3.13 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

## Desktop camera

```bash
python main.py --camera 0 --auto-calibrate
```

Run `python camera_scan.py` to check camera indexes. Open the generated `camera_*.jpg` previews and pass the matching index to `--camera`. Use `--debug` to inspect intermediate processing stages.

Place one flat banknote near the center guide against a plain background. The desktop window shows the feed, segmentation mask, and perspective crop. In the mask, aim for a white banknote on a black background.

| Key | Action |
| --- | --- |
| A | Start guided calibration. Hold the note steady until calibration finishes. |
| C | Cancel calibration. |
| S | Save the current slider settings. |
| T | Toggle printed-number template capture. |
| 2 / 5 / 1 / B / 0 | Save the current crop as a color sample or number template for ฿20 / ฿50 / ฿100 / ฿500 / ฿1,000. |
| Q / Esc | Quit. |

The segmentation sliders adjust saturation, brightness, and morphological cleanup. Increase **S min** or **V min** to suppress low-saturation or dark background regions. Decrease **V max** to suppress glare. Increase **Open** to remove small white specks, or **Close** to fill holes in the note mask.

## Phone camera

```bash
python web_app.py --https --port 5001
```

With the phone and computer on the same network, open `https://<computer-lan-ip>:5001` on the phone. The development server uses a self-signed certificate. Grant the browser camera permission, position the note in the guide, and select **Auto Calibrate**.

## Development

```bash
python -m unittest discover -s tests -v
python test_image.py assets/reference_photos/100_front.jpg
python make_template.py --help
```

The still-image tool exports preprocessing, segmentation, and contour previews to the current directory. When a usable crop is found, it also exports the perspective crop and foreground images. Pass `--help` to see the threshold overrides.

- `thai_banknote/` contains the desktop and Flask apps, the detection pipeline, and the browser UI.
- `assets/` contains front and back reference photos, color histograms, printed-number templates, and the calibration defaults.
- `tests/` covers guided calibration, foreground masking, and Flask endpoints.

The apps share a local `calibration_settings.json` after you save calibration. Teaching color samples or capturing number templates updates the reference data. The repository also recognizes legacy `hist_samples.json` and `note_templates/` at the project root for existing setups.
