"""Tests for traversability-risk evaluation metrics."""

import unittest

import numpy as np

from src.evaluation import confusion_matrix_3x3, evaluate_risk_masks


class EvaluationMetricsTest(unittest.TestCase):
    def test_perfect_prediction_returns_one_for_key_metrics(self):
        gt = np.array([[0, 1, 2], [0, 1, 2]], dtype=np.uint8)
        pred = gt.copy()

        _, metrics = evaluate_risk_masks(gt, pred)

        self.assertEqual(metrics.accuracy, 1.0)
        self.assertEqual(metrics.balanced_accuracy, 1.0)
        self.assertEqual(metrics.macro_f1, 1.0)
        self.assertEqual(metrics.high_risk_recall, 1.0)

    def test_known_synthetic_example_confusion_matrix(self):
        gt = np.array([[0, 0, 1], [1, 2, 2]], dtype=np.uint8)
        pred = np.array([[0, 2, 1], [2, 0, 2]], dtype=np.uint8)

        matrix = confusion_matrix_3x3(gt, pred)

        expected = np.array(
            [
                [1, 0, 1],
                [0, 1, 1],
                [1, 0, 1],
            ],
            dtype=np.int64,
        )
        np.testing.assert_array_equal(matrix, expected)

    def test_unsafe_to_safe_error_is_calculated_correctly(self):
        gt = np.array([[2, 2, 2, 0]], dtype=np.uint8)
        pred = np.array([[0, 1, 2, 0]], dtype=np.uint8)

        _, metrics = evaluate_risk_masks(gt, pred)

        self.assertAlmostEqual(metrics.unsafe_to_safe_error_rate, 1 / 3)

    def test_safe_to_high_risk_rate_is_calculated_correctly(self):
        gt = np.array([[0, 0, 0, 2]], dtype=np.uint8)
        pred = np.array([[2, 1, 0, 2]], dtype=np.uint8)

        _, metrics = evaluate_risk_masks(gt, pred)

        self.assertAlmostEqual(metrics.safe_to_high_risk_rate, 1 / 3)

    def test_mismatched_shapes_raise_clear_error(self):
        gt = np.zeros((2, 2), dtype=np.uint8)
        pred = np.zeros((2, 3), dtype=np.uint8)

        with self.assertRaisesRegex(ValueError, "same shape"):
            evaluate_risk_masks(gt, pred)


if __name__ == "__main__":
    unittest.main()
