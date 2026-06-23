"""Inspect RUGD image-mask pairs and Phase 1 risk-map labels."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_IMAGES_ROOT, DEFAULT_MASKS_ROOT, discover_image_mask_pairs, load_pair
from src.traversability import (
    annotation_rgb_to_class_id,
    class_id_to_risk_mask,
    color_to_class_id_map,
    load_risk_mapping,
    load_rugd_labels,
)


def summarize_sizes(pairs):
    sizes = Counter()
    for pair in pairs:
        with Image.open(pair.image_path) as image_file:
            image_size = image_file.size
        with Image.open(pair.mask_path) as mask_file:
            mask_size = mask_file.size

        sizes[image_size] += 1
        if image_size != mask_size:
            raise ValueError(f"Size mismatch: {pair.image_path} and {pair.mask_path}")
    return sizes


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect the local RUGD dataset.")
    parser.add_argument("--images_root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--masks_root", default=str(DEFAULT_MASKS_ROOT))
    parser.add_argument("--labels", default="config/rugd_labels.yaml")
    parser.add_argument("--risk_mapping", default="config/risk_mapping.yaml")
    parser.add_argument("--subset_size", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    images_root = Path(args.images_root)
    masks_root = Path(args.masks_root)

    pairs = discover_image_mask_pairs(images_root, masks_root)
    subset = discover_image_mask_pairs(
        images_root,
        masks_root,
        subset_size=args.subset_size,
        seed=args.seed,
    )

    labels = load_rugd_labels(args.labels)
    color_to_id = color_to_class_id_map(labels)
    class_to_risk = load_risk_mapping(args.risk_mapping, labels)
    id_to_name = {label.id: label.name for label in labels}

    print("RUGD dataset summary")
    print(f"images_root: {images_root}")
    print(f"masks_root: {masks_root}")
    print(f"valid RGB-mask pairs: {len(pairs)}")
    print()

    print("Five example pairs:")
    for pair in pairs[:5]:
        print(f"  image: {pair.image_path}")
        print(f"  mask:  {pair.mask_path}")
    print()

    print("Image size statistics:")
    for (width, height), count in summarize_sizes(pairs).most_common():
        print(f"  {width}x{height}: {count}")
    print()

    class_counts = Counter()
    risk_counts = Counter()
    for pair in subset:
        _, mask = load_pair(pair)
        class_mask = annotation_rgb_to_class_id(mask, color_to_id)
        risk_mask = class_id_to_risk_mask(class_mask, class_to_risk)

        for class_id, count in zip(*np.unique(class_mask, return_counts=True)):
            class_counts[int(class_id)] += int(count)
        for risk_value, count in zip(*np.unique(risk_mask, return_counts=True)):
            risk_counts[int(risk_value)] += int(count)

    print(f"Class pixel counts for deterministic subset of {len(subset)}:")
    for class_id, count in sorted(class_counts.items()):
        print(f"  {class_id:2d} {id_to_name[class_id]}: {count}")
    print()

    print(f"Risk pixel counts for deterministic subset of {len(subset)}:")
    for risk_value, count in sorted(risk_counts.items()):
        print(f"  risk {risk_value}: {count}")


if __name__ == "__main__":
    main()
