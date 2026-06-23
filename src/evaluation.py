"""Quantitative evaluation for 0/1/2 traversability risk maps."""

from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np


RISK_LABELS = (0, 1, 2)
RISK_NAMES = {
    0: "low_risk",
    1: "medium_risk",
    2: "high_risk",
}


@dataclass(frozen=True)
class RiskMetrics:
    """Pixel-level metrics for traversability risk prediction."""

    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    per_class_precision: dict[str, float]
    per_class_recall: dict[str, float]
    per_class_f1: dict[str, float]
    high_risk_recall: float | None
    unsafe_to_safe_error_rate: float | None
    unsafe_to_medium_error_rate: float | None
    safe_to_high_risk_rate: float | None
    total_pixels: int

    def to_dict(self) -> dict:
        """Return a JSON/CSV-friendly metrics dictionary."""

        return asdict(self)


def validate_risk_mask_pair(gt_mask: np.ndarray, pred_mask: np.ndarray) -> None:
    """Validate shape and values for a pair of risk masks."""

    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            f"Ground-truth and predicted risk masks must have the same shape. "
            f"Got gt={gt_mask.shape}, pred={pred_mask.shape}"
        )

    for name, mask in (("ground-truth", gt_mask), ("predicted", pred_mask)):
        values = set(int(value) for value in np.unique(mask))
        invalid = sorted(values - set(RISK_LABELS))
        if invalid:
            raise ValueError(f"{name} risk mask contains values outside 0, 1, 2: {invalid}")


def confusion_matrix_3x3(gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    """Compute a 3x3 confusion matrix with rows=GT and columns=prediction."""

    validate_risk_mask_pair(gt_mask, pred_mask)
    gt_flat = np.asarray(gt_mask, dtype=np.uint8).reshape(-1)
    pred_flat = np.asarray(pred_mask, dtype=np.uint8).reshape(-1)

    matrix = np.zeros((3, 3), dtype=np.int64)
    for gt_value, pred_value in zip(gt_flat, pred_flat):
        matrix[int(gt_value), int(pred_value)] += 1
    return matrix


def _safe_divide(numerator: float, denominator: float) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def metrics_from_confusion_matrix(matrix: np.ndarray) -> RiskMetrics:
    """Compute aggregate risk metrics from a 3x3 confusion matrix."""

    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (3, 3):
        raise ValueError(f"Confusion matrix must have shape (3, 3). Got {matrix.shape}")

    total = int(matrix.sum())
    if total == 0:
        raise ValueError("Cannot compute metrics from an empty confusion matrix")

    accuracy = float(np.trace(matrix) / total)
    precision: dict[str, float] = {}
    recall: dict[str, float] = {}
    f1: dict[str, float] = {}
    recalls_for_present_classes: list[float] = []

    for label in RISK_LABELS:
        true_positive = float(matrix[label, label])
        predicted_count = float(matrix[:, label].sum())
        gt_count = float(matrix[label, :].sum())

        class_precision = _safe_divide(true_positive, predicted_count)
        class_recall = _safe_divide(true_positive, gt_count)
        if class_precision is None:
            class_precision = 0.0
        if class_recall is None:
            class_recall = 0.0
        if class_precision + class_recall == 0:
            class_f1 = 0.0
        else:
            class_f1 = 2 * class_precision * class_recall / (class_precision + class_recall)

        name = RISK_NAMES[label]
        precision[name] = float(class_precision)
        recall[name] = float(class_recall)
        f1[name] = float(class_f1)
        if gt_count > 0:
            recalls_for_present_classes.append(float(class_recall))

    balanced_accuracy = float(np.mean(recalls_for_present_classes))
    macro_f1 = float(np.mean([f1[RISK_NAMES[label]] for label in RISK_LABELS]))

    gt_high = float(matrix[2, :].sum())
    gt_low = float(matrix[0, :].sum())
    high_risk_recall = _safe_divide(float(matrix[2, 2]), gt_high)
    unsafe_to_safe = _safe_divide(float(matrix[2, 0]), gt_high)
    unsafe_to_medium = _safe_divide(float(matrix[2, 1]), gt_high)
    safe_to_high = _safe_divide(float(matrix[0, 2]), gt_low)

    return RiskMetrics(
        accuracy=accuracy,
        balanced_accuracy=balanced_accuracy,
        macro_f1=macro_f1,
        per_class_precision=precision,
        per_class_recall=recall,
        per_class_f1=f1,
        high_risk_recall=high_risk_recall,
        unsafe_to_safe_error_rate=unsafe_to_safe,
        unsafe_to_medium_error_rate=unsafe_to_medium,
        safe_to_high_risk_rate=safe_to_high,
        total_pixels=total,
    )


def evaluate_risk_masks(gt_mask: np.ndarray, pred_mask: np.ndarray) -> tuple[np.ndarray, RiskMetrics]:
    """Evaluate one pair of 0/1/2 risk masks."""

    matrix = confusion_matrix_3x3(gt_mask, pred_mask)
    return matrix, metrics_from_confusion_matrix(matrix)
