"""Run single-image Phase 1 or Phase 2 prototype outputs."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import DEFAULT_IMAGES_ROOT, DEFAULT_MASKS_ROOT, find_pair, load_pair
from src.models import DEFAULT_SEGFORMER_MODEL, SegFormerADE20K
from src.traversability import (
    ade20k_class_id_to_risk_mask,
    annotation_rgb_to_risk_mask,
    risk_mask_to_color,
)
from src.visualization import (
    make_risk_overlay,
    save_confidence_heatmap_with_colorbar,
    save_four_panel,
    save_segformer_comparison_panel,
)


def load_settings(path: str | Path = "config/settings.yaml") -> dict:
    """Load project settings if present."""

    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def run_ground_truth(args) -> None:
    """Generate Phase 1 ground-truth risk outputs."""

    pair = find_pair(args.sequence, args.filename, args.images_root, args.masks_root)
    image_rgb, annotation_rgb = load_pair(pair)
    risk_mask = annotation_rgb_to_risk_mask(
        annotation_rgb,
        labels_path=args.labels,
        risk_mapping_path=args.risk_mapping,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = Path(args.filename).stem
    risk_path = output_dir / f"{args.sequence}_{stem}_risk_mask.png"
    risk_color_path = output_dir / f"{args.sequence}_{stem}_risk_color.png"
    panel_path = output_dir / f"{args.sequence}_{stem}_phase1_panel.png"

    Image.fromarray(risk_mask).save(risk_path)
    Image.fromarray(risk_mask_to_color(risk_mask)).save(risk_color_path)
    save_four_panel(panel_path, image_rgb, annotation_rgb, risk_mask)

    print(f"Mode: ground_truth")
    print(f"Input image: {pair.image_path}")
    print(f"Input mask: {pair.mask_path}")
    print(f"Saved risk mask: {risk_path}")
    print(f"Saved risk color image: {risk_color_path}")
    print(f"Saved four-panel visualization: {panel_path}")


def print_top_ade20k_labels(class_mask: np.ndarray, id2label: dict[int, str], limit: int = 10) -> None:
    """Print the most common ADE20K predictions in one mask."""

    total = class_mask.size
    counts = Counter(int(value) for value in class_mask.reshape(-1))
    print(f"Top {limit} ADE20K labels predicted in this image:")
    for class_id, count in counts.most_common(limit):
        label = id2label.get(class_id, f"class_{class_id}")
        percentage = count / total * 100
        print(f"  {class_id:3d} {label:32s} {count:7d} px  {percentage:6.2f}%")


def print_mapping_summary(report, class_mask: np.ndarray) -> None:
    """Print how ADE20K labels were mapped or defaulted."""

    total_pixels = class_mask.size
    fallback_pixels = sum(
        int((class_mask == class_id).sum())
        for class_id in report.fallback_high_risk
    )

    print("ADE20K-to-risk mapping summary:")
    print(f"  labels mapped by keyword: {len(report.mapped_by_keyword)}")
    print(f"  labels defaulted to high risk: {len(report.fallback_high_risk)}")
    print(f"  fallback pixels: {fallback_pixels} / {total_pixels} ({fallback_pixels / total_pixels * 100:.2f}%)")

    if report.mapped_by_keyword:
        print("  keyword matches:")
        for class_id, (label_name, risk_value, keyword) in sorted(report.mapped_by_keyword.items()):
            print(f"    {class_id:3d} {label_name:32s} -> risk {risk_value} via '{keyword}'")

    if report.top_unmapped_labels:
        print("  top unmapped labels defaulted to high risk:")
        for class_id, label_name, count in report.top_unmapped_labels[:10]:
            print(f"    {class_id:3d} {label_name:32s} {count:7d} px")


def run_segformer(args) -> None:
    """Run zero-shot ADE20K SegFormer and compare with RUGD-derived GT risk."""

    settings = load_settings(args.settings)
    model_settings = settings.get("models", {})
    model_name = (
        args.segmentation_model
        or settings.get("segmentation_model")
        or model_settings.get("segmentation_model")
        or DEFAULT_SEGFORMER_MODEL
    )
    device = args.device or settings.get("device") or model_settings.get("device", "auto")

    pair = find_pair(args.sequence, args.filename, args.images_root, args.masks_root)
    image_rgb, annotation_rgb = load_pair(pair)
    gt_risk_mask = annotation_rgb_to_risk_mask(
        annotation_rgb,
        labels_path=args.labels,
        risk_mapping_path=args.risk_mapping,
    )

    segmenter = SegFormerADE20K(model_name=model_name, device=device)
    prediction = segmenter.predict(image_rgb)
    predicted_risk_mask, mapping_report = ade20k_class_id_to_risk_mask(
        prediction.class_id_mask,
        prediction.id2label,
        mapping_path=args.ade20k_risk_mapping,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.filename).stem
    prefix = f"{args.sequence}_{stem}"

    rgb_path = output_dir / f"{prefix}_rgb.png"
    gt_risk_color_path = output_dir / f"{prefix}_gt_risk_color.png"
    pred_risk_color_path = output_dir / f"{prefix}_segformer_risk_color.png"
    confidence_path = output_dir / f"{prefix}_segformer_confidence_heatmap.png"
    overlay_path = output_dir / f"{prefix}_segformer_overlay.png"
    panel_path = output_dir / f"{prefix}_segformer_comparison_panel.png"
    pred_mask_path = output_dir / f"{prefix}_segformer_risk_mask.npy"

    Image.fromarray(image_rgb).save(rgb_path)
    Image.fromarray(risk_mask_to_color(gt_risk_mask)).save(gt_risk_color_path)
    Image.fromarray(risk_mask_to_color(predicted_risk_mask)).save(pred_risk_color_path)
    save_confidence_heatmap_with_colorbar(confidence_path, prediction.confidence)
    Image.fromarray(make_risk_overlay(image_rgb, predicted_risk_mask)).save(overlay_path)
    save_segformer_comparison_panel(
        panel_path,
        image_rgb,
        gt_risk_mask,
        predicted_risk_mask,
        prediction.confidence,
    )
    np.save(pred_mask_path, predicted_risk_mask)

    print("Mode: segformer")
    print("Scientific note: zero-shot ADE20K semantic predictions, not RUGD labels.")
    print(f"Model: {prediction.model_name}")
    print(f"Device used: {prediction.device}")
    print(f"Input image: {pair.image_path}")
    print(f"Input mask for GT risk: {pair.mask_path}")
    print_top_ade20k_labels(prediction.class_id_mask, prediction.id2label)
    print_mapping_summary(mapping_report, prediction.class_id_mask)
    print("Saved outputs:")
    print(f"  RGB image: {rgb_path}")
    print(f"  RUGD GT risk color map: {gt_risk_color_path}")
    print(f"  SegFormer predicted risk color map: {pred_risk_color_path}")
    print(f"  SegFormer confidence heatmap: {confidence_path}")
    print(f"  SegFormer predicted overlay: {overlay_path}")
    print(f"  Five-panel comparison: {panel_path}")
    print(f"  Raw predicted 0/1/2 mask: {pred_mask_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a single-image WinterNav-RUGD mode.")
    parser.add_argument("--mode", choices=["ground_truth", "segformer"], default="ground_truth")
    parser.add_argument("--sequence", default="creek")
    parser.add_argument("--filename", default="creek_00001.png")
    parser.add_argument("--output_dir", default="outputs/phase1_example")
    parser.add_argument("--images_root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--masks_root", default=str(DEFAULT_MASKS_ROOT))
    parser.add_argument("--labels", default="config/rugd_labels.yaml")
    parser.add_argument("--risk_mapping", default="config/risk_mapping.yaml")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--ade20k_risk_mapping", default="config/ade20k_risk_mapping.yaml")
    parser.add_argument("--segmentation_model", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.mode == "ground_truth":
        run_ground_truth(args)
    elif args.mode == "segformer":
        run_segformer(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
