import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from thai_banknote.autocalibration import (GuidedCalibration, choose_settings, guide_box,
                             load_settings, mask_quality, save_settings)
from thai_banknote.segmentation import DEFAULT_SETTINGS, segment_banknote
from thai_banknote.preprocessing import camera_warning


def guided_frame():
    frame = np.full((540, 960, 3), 235, np.uint8)
    x1, y1, x2, y2 = guide_box(frame.shape)
    frame[y1:y2, x1:x2] = (50, 150, 50)
    return frame


class AutoCalibrationTests(unittest.TestCase):
    def test_black_source_is_reported_before_motion(self):
        black = np.zeros((540, 960, 3), np.uint8)
        self.assertIsNotNone(camera_warning(black))
        self.assertIsNone(camera_warning(guided_frame()))
        settings, message = choose_settings([guided_frame(), black, black])
        self.assertIsNone(settings)
        self.assertIn("too dark", message)

    def test_stable_note_selects_usable_thresholds(self):
        frame = guided_frame()
        settings, message = choose_settings([frame.copy() for _ in range(8)])
        self.assertIsNotNone(settings, message)
        processed = {"original": frame, "hsv": cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)}
        self.assertGreater(mask_quality(segment_banknote(processed, settings)["mask"]), .8)

    def test_empty_scene_is_rejected(self):
        frame = np.full((540, 960, 3), 235, np.uint8)
        self.assertIsNone(choose_settings([frame] * 8)[0])

    def test_one_poor_frame_does_not_reject_clear_note(self):
        first = guided_frame()
        second = np.full_like(first, 235)
        settings, message = choose_settings([first, second, first])
        self.assertIsNotNone(settings, message)

    def test_collects_multiple_frames_after_countdown(self):
        session = GuidedCalibration()
        session.start(0)
        frame = guided_frame()
        self.assertIsNone(session.update(frame, 2))
        self.assertEqual(len(session.frames), 0)
        with patch("thai_banknote.autocalibration.choose_settings", return_value=(dict(DEFAULT_SETTINGS), "ready")) as choose:
            for index in range(7):
                self.assertIsNone(session.update(frame, 3 + index * .3))
            settings = session.update(frame, 5.1)
            self.assertIsNotNone(settings)
            self.assertEqual(len(choose.call_args.args[0]), 8)
        self.assertFalse(session.active)

    def test_cancel_does_not_search_or_save(self):
        session = GuidedCalibration()
        session.start(0)
        session.cancel()
        with patch("thai_banknote.autocalibration.choose_settings") as choose:
            self.assertIsNone(session.update(guided_frame(), 10))
            choose.assert_not_called()

    def test_roundtrip_and_corrupt_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "settings.json"
            self.assertEqual(load_settings(path), DEFAULT_SETTINGS)
            settings = dict(DEFAULT_SETTINGS, s_min=30, v_min=40)
            save_settings(settings, path)
            self.assertEqual(load_settings(path), settings)
            path.write_text('{"s_min": true}')
            self.assertEqual(load_settings(path), DEFAULT_SETTINGS)
            path.write_text(json.dumps(settings))
            with self.assertRaises(ValueError):
                save_settings(dict(settings, v_min=255, v_max=0), path)
            self.assertEqual(load_settings(path), settings)

    def test_browser_loads_saved_values_and_honours_manual_sliders(self):
        from thai_banknote import web as web_app
        settings = dict(DEFAULT_SETTINGS, s_min=30, v_min=40)
        with patch.object(web_app, "load_settings", return_value=settings):
            page = web_app.app.test_client().get("/")
            self.assertEqual(page.status_code, 200)
            self.assertIn(b'"s_min": 30', page.data)
            self.assertEqual(web_app.settings_from_form({})["v_min"], 40)
            self.assertEqual(web_app.settings_from_form({"v_min": "50"})["v_min"], 50)


if __name__ == "__main__":
    unittest.main()
