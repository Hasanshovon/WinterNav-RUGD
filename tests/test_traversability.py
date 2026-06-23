"""Tests for RUGD color and risk-map conversion."""

import unittest

import numpy as np

from src import traversability


class TraversabilityTest(unittest.TestCase):
    def setUp(self):
        self.labels = traversability.load_rugd_labels()
        self.color_to_id = traversability.color_to_class_id_map(self.labels)
        self.class_to_risk = traversability.load_risk_mapping(labels=self.labels)

    def test_rgb_color_to_class_id_conversion(self):
        mask = np.array(
            [
                [[108, 64, 20], [64, 64, 64]],
                [[255, 229, 204], [255, 0, 0]],
            ],
            dtype=np.uint8,
        )

        class_mask = traversability.annotation_rgb_to_class_id(mask, self.color_to_id)

        np.testing.assert_array_equal(class_mask, np.array([[1, 10], [2, 12]]))

    def test_class_id_to_risk_conversion(self):
        class_mask = np.array([[1, 10], [2, 12]], dtype=np.int16)

        risk_mask = traversability.class_id_to_risk_mask(class_mask, self.class_to_risk)

        np.testing.assert_array_equal(risk_mask, np.array([[0, 0], [1, 2]], dtype=np.uint8))

    def test_unknown_color_detection(self):
        mask = np.array([[[123, 45, 67]]], dtype=np.uint8)

        with self.assertRaises(traversability.UnknownMaskColorError):
            traversability.annotation_rgb_to_class_id(mask, self.color_to_id)

    def test_risk_mask_to_color_shape_and_values(self):
        risk_mask = np.array([[0, 1, 2]], dtype=np.uint8)

        color_mask = traversability.risk_mask_to_color(risk_mask)

        self.assertEqual(color_mask.shape, (1, 3, 3))
        np.testing.assert_array_equal(color_mask[0, 0], [0, 255, 0])
        np.testing.assert_array_equal(color_mask[0, 1], [255, 255, 0])
        np.testing.assert_array_equal(color_mask[0, 2], [255, 0, 0])


if __name__ == "__main__":
    unittest.main()
