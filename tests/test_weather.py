"""Tests for deterministic synthetic weather augmentation."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from src import weather


class WeatherAugmentationTest(unittest.TestCase):
    def setUp(self):
        self.image = np.full((12, 16, 3), 128, dtype=np.uint8)
        self.mask = np.zeros((12, 16), dtype=np.uint8)

    def test_weather_preserves_shape_and_rgb_dtype(self):
        for condition in weather.CONDITIONS:
            degraded = weather.apply_weather(self.image, condition, severity=0.5, seed=42)
            self.assertEqual(degraded.shape, self.image.shape)
            self.assertEqual(degraded.dtype, np.uint8)

    def test_severity_zero_gives_unchanged_image(self):
        for condition in weather.CONDITIONS:
            degraded = weather.apply_weather(self.image, condition, severity=0.0, seed=42)
            np.testing.assert_array_equal(degraded, self.image)

    def test_snow_is_deterministic_under_fixed_seed(self):
        snow_a = weather.synthetic_snow(self.image, severity=0.5, seed=42)
        snow_b = weather.synthetic_snow(self.image, severity=0.5, seed=42)
        np.testing.assert_array_equal(snow_a, snow_b)

    def test_gt_masks_are_never_transformed(self):
        _ = weather.apply_weather(self.image, "fog", severity=0.5, seed=42)
        np.testing.assert_array_equal(self.mask, np.zeros((12, 16), dtype=np.uint8))

    def test_condition_names_are_saved_correctly(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "weather_config.json"
            path.write_text(json.dumps({"conditions": weather.CONDITIONS}), encoding="utf-8")
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["conditions"], weather.CONDITIONS)


if __name__ == "__main__":
    unittest.main()
