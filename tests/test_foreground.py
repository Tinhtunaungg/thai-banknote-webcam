"""Run with: python -m unittest test_foreground -v (no camera required)."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from thai_banknote.classifier import compute_signature
from thai_banknote.foreground import refine_banknote


class ForegroundTests(unittest.TestCase):
    def test_skin_coloured_note_does_not_disappear(self):
        image = np.full((300, 640, 3), (100, 140, 190), np.uint8)
        result = refine_banknote(image)
        self.assertTrue(result["usable"])
        self.assertIn("ambiguous skin hints ignored", result["status"])
        self.assertIsNotNone(compute_signature(image, mask=result["mask"]))

    def test_border_connected_obstruction(self):
        # A green note with a skin-coloured object entering from the right.
        image = np.full((300, 640, 3), (60, 160, 60), np.uint8)
        cv2.rectangle(image, (490, 60), (639, 230), (100, 140, 190), -1)
        result = refine_banknote(image)
        self.assertTrue(result["usable"])
        self.assertLess(np.mean(result["mask"][90:200, 520:620] > 0), 0.1)
        self.assertGreater(np.mean(result["mask"][100:200, 100:400] > 0), 0.8)
        self.assertEqual(result["mask"].shape, image.shape[:2])

    def test_mask_excludes_colours_from_histogram(self):
        image = np.full((100, 200, 3), (0, 255, 0), np.uint8)
        mask = np.full((100, 200), 255, np.uint8)
        mask[:, 100:] = 0
        expected = compute_signature(image, mask=mask)
        image[:, 100:] = (0, 0, 255)
        np.testing.assert_array_equal(compute_signature(image, mask=mask), expected)
        self.assertFalse(np.array_equal(compute_signature(image), expected))

    def test_insufficient_visible_pixels_reject_signature(self):
        image = np.full((100, 200, 3), 100, np.uint8)
        self.assertIsNone(compute_signature(image, mask=np.zeros((100, 200), np.uint8)))

    def test_missing_crop_and_failed_grabcut_reject(self):
        self.assertFalse(refine_banknote(None)["usable"])
        image = np.full((100, 200, 3), (0, 255, 0), np.uint8)
        with patch("thai_banknote.foreground.cv2.grabCut", side_effect=cv2.error("failure")):
            result = refine_banknote(image)
        self.assertFalse(result["usable"])
        self.assertIsNone(compute_signature(image, mask=result["mask"]))

    def test_disabled_stage_preserves_original_signature(self):
        image = np.full((100, 200, 3), (0, 255, 0), np.uint8)
        result = refine_banknote(image, enabled=False)
        self.assertEqual(result["coverage"], 1.0)
        np.testing.assert_array_equal(compute_signature(image),
                                      compute_signature(image, mask=result["mask"]))
        np.testing.assert_array_equal(result["preview"], image)

    def test_browser_endpoint_and_debug_mask(self):
        from thai_banknote import web as web_app

        image = np.full((300, 640, 3), (60, 160, 60), np.uint8)
        _, jpeg = cv2.imencode(".jpg", image)
        import io
        client = web_app.app.test_client()
        with patch.object(web_app, "crop_banknote", return_value=(image, None)), \
                patch.object(web_app, "find_banknote_contours", return_value=[{"test": True}]):
            response = client.post("/process_frame", data={
                "frame": (io.BytesIO(jpeg.tobytes()), "frame.jpg"), "include_mask": "1"})
            self.assertEqual(response.status_code, 200)
            result = response.get_json()
            self.assertEqual(result["foreground_status"], "refined")
            self.assertTrue(result["foreground_mask"].startswith("data:image/png;base64,"))
            response = client.post("/process_frame", data={
                "frame": (io.BytesIO(jpeg.tobytes()), "frame.jpg"), "use_grabcut": "0"})
            self.assertEqual(response.get_json()["foreground_status"], "disabled")


if __name__ == "__main__":
    unittest.main()
