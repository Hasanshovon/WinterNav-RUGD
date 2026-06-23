"""Tests for ADE20K-to-risk keyword mapping."""

import unittest

import numpy as np

from src.traversability import ade20k_class_id_to_risk_mask


class Ade20kRiskMappingTest(unittest.TestCase):
    def test_keyword_mapping_produces_expected_risk_values(self):
        class_mask = np.array([[1, 2, 3, 4]], dtype=np.int16)
        id2label = {
            1: "road",
            2: "grass",
            3: "tree",
            4: "signboard",
        }

        risk_mask, report = ade20k_class_id_to_risk_mask(class_mask, id2label)

        np.testing.assert_array_equal(risk_mask, np.array([[0, 1, 2, 2]], dtype=np.uint8))
        self.assertEqual(report.mapped_by_keyword[1], ("road", 0, "road"))
        self.assertEqual(report.mapped_by_keyword[2], ("grass", 1, "grass"))
        self.assertEqual(report.mapped_by_keyword[3], ("tree", 2, "tree"))
        self.assertEqual(report.mapped_by_keyword[4], ("signboard", 2, "sign"))

    def test_unmapped_labels_default_to_high_risk(self):
        class_mask = np.array([[99]], dtype=np.int16)
        id2label = {99: "ceiling"}

        risk_mask, report = ade20k_class_id_to_risk_mask(class_mask, id2label)

        np.testing.assert_array_equal(risk_mask, np.array([[2]], dtype=np.uint8))
        self.assertEqual(report.fallback_high_risk, {99: "ceiling"})


if __name__ == "__main__":
    unittest.main()
