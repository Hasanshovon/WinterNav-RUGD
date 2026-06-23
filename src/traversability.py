"""Convert RUGD RGB annotation masks into traversability risk maps."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np
import yaml


LOW_RISK_COLOR = np.array([0, 255, 0], dtype=np.uint8)
MEDIUM_RISK_COLOR = np.array([255, 255, 0], dtype=np.uint8)
HIGH_RISK_COLOR = np.array([255, 0, 0], dtype=np.uint8)


@dataclass(frozen=True)
class RugdLabel:
    """One verified RUGD semantic label."""

    id: int
    name: str
    color: tuple[int, int, int]


class UnknownMaskColorError(ValueError):
    """Raised when an annotation mask contains RGB colors outside the colormap."""

    def __init__(self, unknown_colors: list[tuple[int, int, int]]):
        self.unknown_colors = unknown_colors
        colors = ", ".join(str(color) for color in unknown_colors[:10])
        super().__init__(f"Unknown RGB colors in annotation mask: {colors}")


@dataclass(frozen=True)
class Ade20kMappingReport:
    """Summary of ADE20K label-name risk mapping for one prediction."""

    mapped_by_keyword: dict[int, tuple[str, int, str]]
    fallback_high_risk: dict[int, str]
    top_unmapped_labels: list[tuple[int, str, int]]


def load_rugd_labels(config_path: str | Path = "config/rugd_labels.yaml") -> list[RugdLabel]:
    """Load verified RUGD labels from YAML."""

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    labels = []
    for item in data.get("labels", []):
        color = tuple(int(value) for value in item["color"])
        if len(color) != 3:
            raise ValueError(f"Label {item['name']} does not have an RGB color")
        labels.append(
            RugdLabel(id=int(item["id"]), name=str(item["name"]), color=color)
        )

    if not labels:
        raise ValueError(f"No RUGD labels found in {config_path}")

    return labels


def color_to_class_id_map(labels: list[RugdLabel]) -> dict[tuple[int, int, int], int]:
    """Build an RGB-color to class-ID lookup from verified labels."""

    return {label.color: label.id for label in labels}


def label_name_to_id_map(labels: list[RugdLabel]) -> dict[str, int]:
    """Build a label-name to class-ID lookup."""

    return {label.name: label.id for label in labels}


def load_risk_mapping(
    risk_mapping_path: str | Path = "config/risk_mapping.yaml",
    labels: list[RugdLabel] | None = None,
) -> dict[int, int]:
    """Load class-ID to risk-label mapping from YAML."""

    labels = labels if labels is not None else load_rugd_labels()
    name_to_id = label_name_to_id_map(labels)
    risk_mapping_path = Path(risk_mapping_path)

    with risk_mapping_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    class_to_risk: dict[int, int] = {}
    for risk_level in data.get("risk_levels", {}).values():
        risk_value = int(risk_level["value"])
        if risk_value not in {0, 1, 2}:
            raise ValueError(f"Risk value must be 0, 1, or 2. Got {risk_value}")
        for label_name in risk_level.get("labels", []):
            if label_name not in name_to_id:
                raise ValueError(f"Risk mapping references unknown label: {label_name}")
            class_to_risk[name_to_id[label_name]] = risk_value

    expected_ids = {label.id for label in labels}
    missing_ids = sorted(expected_ids - set(class_to_risk))
    if missing_ids:
        raise ValueError(f"Risk mapping is missing class IDs: {missing_ids}")

    return class_to_risk


def annotation_rgb_to_class_id(
    annotation_rgb: np.ndarray,
    color_to_id: dict[tuple[int, int, int], int],
) -> np.ndarray:
    """Convert an RGB RUGD annotation mask to a class-ID mask."""

    if annotation_rgb.ndim != 3 or annotation_rgb.shape[2] != 3:
        raise ValueError("annotation_rgb must have shape (height, width, 3)")

    annotation_rgb = np.asarray(annotation_rgb, dtype=np.uint8)
    unique_colors = np.unique(annotation_rgb.reshape(-1, 3), axis=0)
    unknown_colors = [
        tuple(int(value) for value in color)
        for color in unique_colors
        if tuple(int(value) for value in color) not in color_to_id
    ]
    if unknown_colors:
        raise UnknownMaskColorError(unknown_colors)

    class_mask = np.full(annotation_rgb.shape[:2], -1, dtype=np.int16)
    for color, class_id in color_to_id.items():
        color_array = np.array(color, dtype=np.uint8)
        class_mask[np.all(annotation_rgb == color_array, axis=2)] = class_id

    return class_mask


def class_id_to_risk_mask(
    class_id_mask: np.ndarray,
    class_to_risk: dict[int, int],
) -> np.ndarray:
    """Convert a RUGD class-ID mask to a 0/1/2 traversability risk mask."""

    class_id_mask = np.asarray(class_id_mask)
    risk_mask = np.full(class_id_mask.shape, 255, dtype=np.uint8)

    for class_id, risk_value in class_to_risk.items():
        risk_mask[class_id_mask == class_id] = risk_value

    unknown_ids = sorted(int(value) for value in np.unique(class_id_mask[risk_mask == 255]))
    if unknown_ids:
        raise ValueError(f"No risk mapping found for class IDs: {unknown_ids}")

    return risk_mask


def annotation_rgb_to_risk_mask(
    annotation_rgb: np.ndarray,
    labels_path: str | Path = "config/rugd_labels.yaml",
    risk_mapping_path: str | Path = "config/risk_mapping.yaml",
) -> np.ndarray:
    """Convert an RGB RUGD annotation mask directly to a risk mask."""

    labels = load_rugd_labels(labels_path)
    class_mask = annotation_rgb_to_class_id(annotation_rgb, color_to_class_id_map(labels))
    return class_id_to_risk_mask(class_mask, load_risk_mapping(risk_mapping_path, labels))


def risk_mask_to_color(risk_mask: np.ndarray) -> np.ndarray:
    """Convert a 0/1/2 risk mask to green/yellow/red RGB colors."""

    risk_mask = np.asarray(risk_mask)
    color_mask = np.zeros((*risk_mask.shape, 3), dtype=np.uint8)

    color_mask[risk_mask == 0] = LOW_RISK_COLOR
    color_mask[risk_mask == 1] = MEDIUM_RISK_COLOR
    color_mask[risk_mask == 2] = HIGH_RISK_COLOR

    unknown_values = sorted(int(value) for value in np.unique(risk_mask) if value not in {0, 1, 2})
    if unknown_values:
        raise ValueError(f"Risk mask contains values outside 0, 1, 2: {unknown_values}")

    return color_mask


def normalize_label_name(label: str) -> str:
    """Normalize semantic label names for simple keyword matching."""

    return re.sub(r"[^a-z0-9]+", " ", label.lower()).strip()


def load_ade20k_risk_keywords(
    config_path: str | Path = "config/ade20k_risk_mapping.yaml",
) -> tuple[dict[int, list[str]], int]:
    """Load ADE20K keyword rules and the default fallback risk."""

    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    keywords_by_risk: dict[int, list[str]] = {}
    for risk_level in data.get("risk_levels", {}).values():
        risk_value = int(risk_level["value"])
        keywords_by_risk[risk_value] = [
            normalize_label_name(keyword)
            for keyword in risk_level.get("keywords", [])
        ]

    return keywords_by_risk, int(data.get("default_risk", 2))


def match_ade20k_label_to_risk(
    label_name: str,
    keywords_by_risk: dict[int, list[str]],
    default_risk: int = 2,
) -> tuple[int, str | None]:
    """Map one ADE20K label name to a risk value with normalized keywords."""

    normalized_label = normalize_label_name(label_name)
    words = set(normalized_label.split())

    for risk_value in (0, 1, 2):
        for keyword in keywords_by_risk.get(risk_value, []):
            keyword_words = keyword.split()
            if normalized_label == keyword:
                return risk_value, keyword
            if len(keyword_words) == 1 and (keyword in words or keyword in normalized_label):
                return risk_value, keyword
            if len(keyword_words) > 1 and keyword in normalized_label:
                return risk_value, keyword

    return default_risk, None


def describe_ade20k_risk_mapping(
    ade20k_class_mask: np.ndarray,
    id2label: dict[int, str],
    mapping_path: str | Path = "config/ade20k_risk_mapping.yaml",
) -> Ade20kMappingReport:
    """Report keyword matches and high-risk fallbacks for ADE20K labels in a mask."""

    keywords_by_risk, default_risk = load_ade20k_risk_keywords(mapping_path)
    mapped_by_keyword: dict[int, tuple[str, int, str]] = {}
    fallback_high_risk: dict[int, str] = {}

    unique_ids, counts = np.unique(ade20k_class_mask, return_counts=True)
    count_by_id = {int(class_id): int(count) for class_id, count in zip(unique_ids, counts)}

    for class_id in sorted(count_by_id):
        label_name = id2label.get(class_id, f"class_{class_id}")
        risk_value, keyword = match_ade20k_label_to_risk(
            label_name,
            keywords_by_risk,
            default_risk,
        )
        if keyword is None:
            fallback_high_risk[class_id] = label_name
        else:
            mapped_by_keyword[class_id] = (label_name, risk_value, keyword)

    top_unmapped = sorted(
        (
            (class_id, label_name, count_by_id[class_id])
            for class_id, label_name in fallback_high_risk.items()
        ),
        key=lambda item: item[2],
        reverse=True,
    )

    return Ade20kMappingReport(
        mapped_by_keyword=mapped_by_keyword,
        fallback_high_risk=fallback_high_risk,
        top_unmapped_labels=top_unmapped,
    )


def ade20k_class_id_to_risk_mask(
    ade20k_class_mask: np.ndarray,
    id2label: dict[int, str],
    mapping_path: str | Path = "config/ade20k_risk_mapping.yaml",
) -> tuple[np.ndarray, Ade20kMappingReport]:
    """Convert ADE20K class IDs to 0/1/2 traversability risk labels.

    Unmapped ADE20K labels default to high risk. This is a heuristic
    zero-shot mapping and does not convert ADE20K classes into RUGD classes.
    """

    keywords_by_risk, default_risk = load_ade20k_risk_keywords(mapping_path)
    ade20k_class_mask = np.asarray(ade20k_class_mask)
    risk_mask = np.full(ade20k_class_mask.shape, default_risk, dtype=np.uint8)

    for class_id in np.unique(ade20k_class_mask):
        class_id = int(class_id)
        label_name = id2label.get(class_id, f"class_{class_id}")
        risk_value, _ = match_ade20k_label_to_risk(
            label_name,
            keywords_by_risk,
            default_risk,
        )
        risk_mask[ade20k_class_mask == class_id] = risk_value

    report = describe_ade20k_risk_mapping(ade20k_class_mask, id2label, mapping_path)
    return risk_mask, report
