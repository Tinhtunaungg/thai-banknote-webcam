import argparse

import cv2
import sys


def open_capture(index):
    # macOS needs AVFoundation; Windows works best with DirectShow; Linux uses the default.
    if sys.platform == "darwin":
        return cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if sys.platform.startswith("win"):
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def parse_args():
    parser = argparse.ArgumentParser(description="Find available OpenCV camera indexes.")
    parser.add_argument("--max-index", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def open_camera(index, width, height):
    cap = open_capture(index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def main():
    args = parse_args()
    found = []

    for index in range(args.max_index + 1):
        cap = open_camera(index, args.width, args.height)

        if not cap.isOpened():
            print(f"camera {index}: not available")
            cap.release()
            continue

        ok, frame = cap.read()
        if not ok or frame is None:
            print(f"camera {index}: opened but no frame")
            cap.release()
            continue

        output_path = f"camera_{index}.jpg"
        cv2.imwrite(output_path, frame)
        height, width = frame.shape[:2]
        print(f"camera {index}: OK {width}x{height}, saved {output_path}")
        found.append(index)
        cap.release()

    if found:
        print(f"available camera indexes: {found}")
        print("Open the saved camera_*.jpg files to find your iPhone camera.")
    else:
        print("No cameras found by OpenCV.")


if __name__ == "__main__":
    main()
