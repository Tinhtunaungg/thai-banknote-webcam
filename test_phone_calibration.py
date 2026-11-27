import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from autocalibration import guide_box, load_settings, save_settings
import web_app


def scene():
    image = np.full((540, 960, 3), 235, np.uint8)
    x1, y1, x2, y2 = guide_box(image.shape)
    image[y1:y2, x1:x2] = (50, 150, 50)
    return image


def upload_frames(image, count=8):
    _, encoded = cv2.imencode(".jpg", image)
    return {"frames": [(io.BytesIO(encoded.tobytes()), f"{i}.jpg") for i in range(count)]}


class PhoneCalibrationTests(unittest.TestCase):
    def setUp(self):
        self.client = web_app.app.test_client()

    def test_success_saves_settings_and_next_detection_uses_them(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "phone.json"
            with patch("web_app.save_settings", side_effect=lambda s: save_settings(s, path)):
                response = self.client.post("/auto_calibrate", data=upload_frames(scene()))
            self.assertEqual(response.status_code, 200)
            result = response.get_json()
            self.assertEqual(result["settings"], load_settings(path))
            self.assertTrue(result["mask"].startswith("data:image/png;base64,"))
            image = scene()
            _, encoded = cv2.imencode(".jpg", image)
            with patch("web_app.load_settings", return_value=load_settings(path)):
                detected = self.client.post("/process_frame", data={
                    "frame": (io.BytesIO(encoded.tobytes()), "note.jpg")})
            self.assertEqual(detected.status_code, 200)
            self.assertNotEqual(detected.get_json()["foreground_status"], "no crop")

    def test_rejects_missing_and_unreadable_frames(self):
        self.assertEqual(self.client.post("/auto_calibrate", data=upload_frames(scene(), 7)).status_code, 400)
        data = {"frames": [(io.BytesIO(b""), f"{i}.jpg") for i in range(8)]}
        self.assertEqual(self.client.post("/auto_calibrate", data=data).status_code, 400)

    def test_dark_or_empty_scene_keeps_previous_settings(self):
        for value in (0, 235):
            image = np.full((540, 960, 3), value, np.uint8)
            with patch("web_app.save_settings") as save:
                response = self.client.post("/auto_calibrate", data=upload_frames(image))
                self.assertEqual(response.status_code, 422)
                self.assertTrue(response.get_json()["kept_previous_settings"])
                self.assertNotIn("press A", response.get_json()["error"])
                save.assert_not_called()

    def test_save_error_is_reported(self):
        with patch("web_app.save_settings", side_effect=OSError("test")), \
                patch.object(web_app.app.logger, "exception"):
            response = self.client.post("/auto_calibrate", data=upload_frames(scene()))
        self.assertEqual(response.status_code, 503)

    def test_page_has_auto_control_and_no_manual_sliders(self):
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"Auto Calibrate", response.data)
        self.assertIn(b'id="calibration-guide"', response.data)
        self.assertNotIn(b'type="range"', response.data)
        self.assertNotIn(b"wireSlider", response.data)


if __name__ == "__main__":
    unittest.main()
