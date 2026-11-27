# Friend version with added processing stages

This folder combines the friend's printed-number template matching and
cross-platform camera handling with GrabCut, masked colour histograms,
automatic calibration, camera warnings, and on-screen instructions. The
phone page uses Auto Calibrate instead of manual sliders. The friend's colour
samples, number templates and reference photos are preserved.

Phone server (stop the old server first):

```bash
cd "/Users/skiz_tha/Documents/New project/review-friend-update-2026-10-12/thai-banknote-webcam"
"/Users/skiz_tha/Documents/New project/thai-banknote-webcam/.venv/bin/python" web_app.py --https --port 5001
```

Desktop with iPhone camera 1:

```bash
"/Users/skiz_tha/Documents/New project/thai-banknote-webcam/.venv/bin/python" main.py --camera 1 --debug --auto-calibrate
```

The commands reuse the original folder's Python environment but run the
friend's merged source. The original folder is separate.

A starts calibration, C cancels it, S saves manual desktop settings, and Q
quits. T toggles number-template capture; then use 2/5/1/b/0 to label a selected
number template. Outside template mode these keys save colour samples.
Calibration remains a single live-frame capture that chooses and saves the
five settings, not continuous adaptation. Centre a complete note near the
phone guide; exact guide fit is no longer required.

GrabCut refines the crop and excludes masked pixels from colour recognition.
Number matching still uses the original crop to preserve printed details;
the friend's shape-only and whole-frame fallback remain available on the web.
Those fallback decisions may be made without a usable GrabCut mask.
The supplied templates now cover all five denominations: 20, 50, 100, 500
and 1000 baht. Four 1000-baht number templates and front/back reference photos
were added from thai-banknote-webcam_add1000.zip.

No camera, browser or automated tests were run for this merge, as requested
by the user. Calibration and GrabCut need evaluation on the actual camera.
The pre-change friend folder is preserved in a verified backup ZIP in the
workspace root.

---

# Thai Banknote OpenCV Project

We are starting simple.

Current goal:

```text
Webcam
  -> Capture Frame
  -> Preprocessing
  -> Segment the Banknote
  -> Find Contour Candidate
  -> Perspective Crop
  -> Denomination Color Classification
  -> Single Banknote Total
```

Multiple-banknote counting is not implemented yet.

## Current Pipeline

```text
webcam frame
  -> resize to 960x540 for stable display
  -> grayscale
  -> Gaussian blur
  -> histogram equalization for contrast
  -> HSV conversion
  -> saturation + brightness mask
  -> morphological cleanup
  -> show segmented banknote
  -> find contours from mask
  -> draw largest banknote candidate
  -> rotated rectangle filtering
  -> perspective crop to flat banknote image
  -> white-balance crop using full-frame gray-world gains
  -> compute Hue-Saturation histogram from crop center
  -> classify by best histogram correlation against learned samples
  -> display detected denomination and single-note total
```

## Setup

```bash
cd "/Users/skiz_tha/Documents/New project/thai-banknote-webcam"
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

First find your camera index:

```bash
python3 camera_scan.py
```

This saves files like:

```text
camera_0.jpg
camera_1.jpg
camera_2.jpg
```

Open those images and find the one from your iPhone camera. Use that number with `--camera`.

```bash
python3 main.py --camera 0
```

For example, if `camera_2.jpg` is your iPhone:

```bash
python3 main.py --camera 2
```

Press `q` or `Esc` to quit.

## Windows You Should See

Normal mode shows one window, `Thai Banknote Detector`:

- top: camera feed + detection result
- bottom-left: `2 - Segmentation Mask`
- bottom-right: `3 - Perspective Crop`

For extra intermediate processing, run:

```bash
python3 main.py --camera 2 --debug
```

Debug mode also shows:

- `Debug - Gray`
- `Debug - Blurred`
- `Debug - Segmented Banknote`
- `Debug - Contour Candidate`

## Which Window Needs Calibration?

The trackbars sit on top of the `Thai Banknote Detector` window. Watch the bottom-left `2 - Segmentation Mask` panel while you move them.

The goal is:

- banknote = white
- background = black

Use `--debug` only if you want to inspect extra intermediate processing windows.

## Slider Guide

The sliders on the `Thai Banknote Detector` window control the segmentation mask panel:

- `S min`: minimum color saturation. Increase it if the background is white in the mask. Decrease it if the banknote disappears.
- `V min`: minimum brightness. Increase it to remove dark background/shadows. Decrease it if dark parts of the note disappear.
- `V max`: maximum brightness. Decrease it to remove glare or very bright background.
- `Open`: removes small white noise. Increase if the mask has many small dots.
- `Close`: fills small black holes inside the banknote. Increase if the banknote mask is broken.

Current default values are from your working iPhone camera calibration:

```text
S min: 0
V min: 247
V max: 255
Open: 0
Close: 5
```

For segmentation, check:

- the banknote is clearly visible in the captured frame
- the banknote area is mostly white in the segmentation mask panel
- the background is mostly black in the segmentation mask panel

If the background is also white in the mask, use a plainer background.
If the banknote disappears, improve lighting or lower the saturation threshold later.

For perspective crop, check:

- `8 - Rotated Rectangle` should outline the banknote with a magenta box.
- `3 - Perspective Crop` should show a flat rectangular crop of the banknote.
- If the crop is tilted, incomplete, or includes too much background, improve the segmentation mask first.

## Teach Denominations

Use one banknote at a time. Wait until `3 - Perspective Crop` is stable, then press:

- `2` to save the current crop color as `20 baht`
- `5` to save the current crop color as `50 baht`
- `1` to save the current crop color as `100 baht`
- `b` to save the current crop color as `500 baht`
- `0` to save the current crop color as `1000 baht`

The learned histogram samples are saved in:

```text
hist_samples.json
```

Take 3-5 samples per denomination under your normal lighting. Move the note slightly between samples so the match is more robust.

After learning samples, `1 - Camera + Detection` shows:

- detected denomination
- distance score
- total for the single visible banknote

## Test One Image

```bash
python3 test_image.py path/to/photo.jpg
```

You can also test calibration values:

```bash
python3 test_image.py path/to/photo.jpg --s-min 45 --v-min 50 --v-max 240 --open 1 --close 2
```

This saves:

- `preprocess_original.jpg`
- `preprocess_gray.jpg`
- `preprocess_blurred.jpg`
- `segment_mask.jpg`
- `segment_banknote.jpg`
- `contour_candidate.jpg`
- `rotated_rectangle.jpg`
- `perspective_crop.jpg`

## Next Step Later

After this works well, the next stage will be:

```text
Single Banknote Total
  -> Multiple Banknote Counting
```
