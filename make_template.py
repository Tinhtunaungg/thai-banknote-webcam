import argparse

import cv2

from perspective import CROP_HEIGHT, CROP_WIDTH
from template_matcher import SLUG_TO_LABEL, save_template

WINDOW = "Drag a TIGHT box around the number, then ENTER (c = cancel)"


def main():
    parser = argparse.ArgumentParser(description="Cut a number template from a banknote image.")
    parser.add_argument("image", help="Image of ONE full banknote, cropped to its edges")
    parser.add_argument("value", choices=list(SLUG_TO_LABEL), help="20, 50, 100, 500 or 1000")
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise SystemExit(f"Could not read image: {args.image}")

    # Same size as the live perspective crop, so the template scale matches.
    note = cv2.resize(image, (CROP_WIDTH, CROP_HEIGHT))

    roi = cv2.selectROI(WINDOW, note, showCrosshair=True)
    cv2.destroyAllWindows()

    path = save_template(note, roi, SLUG_TO_LABEL[args.value])
    print("cancelled" if path is None else f"saved {path}")


if __name__ == "__main__":
    main()