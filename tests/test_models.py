"""Tests for model-output helpers that do not require downloading weights."""

import unittest

import numpy as np

from src.models import (
    MODEL_REGISTRY,
    confidence_map_in_unit_interval,
    prediction_matches_original_size,
    resize_discrete_class_mask_nearest,
    resolve_segmentation_model_name,
)


class ModelHelperTest(unittest.TestCase):
    def test_discrete_class_mask_resize_uses_nearest_neighbor(self):
        class_mask = np.array([[1, 2], [3, 4]], dtype=np.int16)

        resized = resize_discrete_class_mask_nearest(class_mask, size=(4, 4))

        expected = np.array(
            [
                [1, 1, 2, 2],
                [1, 1, 2, 2],
                [3, 3, 4, 4],
                [3, 3, 4, 4],
            ],
            dtype=np.int16,
        )
        np.testing.assert_array_equal(resized, expected)

    def test_confidence_map_stays_within_unit_interval(self):
        confidence = np.array([[0.0, 0.5, 1.0]], dtype=np.float32)

        self.assertTrue(confidence_map_in_unit_interval(confidence))
        self.assertFalse(confidence_map_in_unit_interval(np.array([[1.1]], dtype=np.float32)))

    def test_registry_finds_mask2former_swin_small(self):
        self.assertEqual(
            MODEL_REGISTRY["mask2former_swin_small"],
            "facebook/mask2former-swin-small-ade-semantic",
        )

    def test_registry_finds_upernet_convnext_tiny(self):
        self.assertEqual(
            MODEL_REGISTRY["upernet_convnext_tiny"],
            "openmmlab/upernet-convnext-tiny",
        )

    def test_output_class_mask_has_original_height_width(self):
        class_mask = np.zeros((5, 7), dtype=np.int16)
        confidence = np.ones((5, 7), dtype=np.float32)

        self.assertTrue(
            prediction_matches_original_size(class_mask, confidence, (5, 7, 3))
        )
        self.assertFalse(
            prediction_matches_original_size(class_mask, confidence, (7, 5, 3))
        )

    def test_unknown_model_name_raises_clear_error(self):
        with self.assertRaisesRegex(ValueError, "Unknown model name"):
            resolve_segmentation_model_name("not_a_registered_model")


if __name__ == "__main__":
    unittest.main()
