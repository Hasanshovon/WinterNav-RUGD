"""Run subset-level experiments for WinterNav-RUGD."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
import time
import traceback

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import (
    DEFAULT_IMAGES_ROOT,
    DEFAULT_MASKS_ROOT,
    discover_image_mask_pairs,
    find_pair,
    RugdPair,
    load_pair,
)
from src.evaluation import evaluate_risk_masks, metrics_from_confusion_matrix
from src.models import (
    DEFAULT_SEGFORMER_MODEL,
    MODEL_REGISTRY,
    clear_model_caches,
    create_ade20k_segmenter,
    resolve_segformer_model_name,
)
from src.traversability import ade20k_class_id_to_risk_mask, annotation_rgb_to_risk_mask
from src.visualization import (
    save_failure_analysis_panel,
    save_metric_bar_chart,
    save_segformer_comparison_panel,
    save_weather_comparison_panel,
    save_weather_rows_figure,
)
from src.weather import CONDITIONS, apply_weather


FAILURE_ANALYSIS_IMAGES = [
    ("creek", "creek_01221.png"),
    ("creek", "creek_01086.png"),
    ("creek", "creek_01021.png"),
]
DEFAULT_SELECTED_PAIRS_CSV = Path("outputs/phase3_eval/selected_pairs.csv")
WEATHER_SEVERITY = 0.5


def load_settings(path: str | Path = "config/settings.yaml") -> dict:
    """Load project settings if present."""

    path = Path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}


def flatten_metrics(metrics) -> dict:
    """Flatten nested metric fields for CSV rows."""

    row = {
        "accuracy": metrics.accuracy,
        "balanced_accuracy": metrics.balanced_accuracy,
        "macro_f1": metrics.macro_f1,
        "high_risk_recall": metrics.high_risk_recall,
        "unsafe_to_safe_error_rate": metrics.unsafe_to_safe_error_rate,
        "unsafe_to_medium_error_rate": metrics.unsafe_to_medium_error_rate,
        "safe_to_high_risk_rate": metrics.safe_to_high_risk_rate,
        "total_pixels": metrics.total_pixels,
    }
    for class_name, value in metrics.per_class_precision.items():
        row[f"precision_{class_name}"] = value
    for class_name, value in metrics.per_class_recall.items():
        row[f"recall_{class_name}"] = value
    for class_name, value in metrics.per_class_f1.items():
        row[f"f1_{class_name}"] = value
    return row


def write_csv(path: Path, rows: list[dict], fieldnames: list[str] | None = None) -> None:
    """Write rows to CSV, creating an empty file with headers when needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = sorted({key for row in rows for key in row})

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_confusion_matrix_csv(path: Path, matrix: np.ndarray) -> None:
    """Write a labelled 3x3 confusion matrix CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["gt\\pred", "0_low", "1_medium", "2_high"])
        for index, row_name in enumerate(["0_low", "1_medium", "2_high"]):
            writer.writerow([row_name, *[int(value) for value in matrix[index]]])


def cuda_memory_summary() -> dict[str, int | str]:
    """Return current CUDA memory usage if CUDA is available."""

    if not torch.cuda.is_available():
        return {"device": "cpu"}
    device_index = torch.cuda.current_device()
    return {
        "device": torch.cuda.get_device_name(device_index),
        "allocated_bytes": int(torch.cuda.memory_allocated(device_index)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device_index)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device_index)),
    }


def print_cuda_memory(prefix: str) -> None:
    """Print CUDA memory usage when available."""

    memory = cuda_memory_summary()
    if memory.get("device") == "cpu":
        print(f"{prefix}: CUDA not available")
        return
    print(
        f"{prefix}: allocated={memory['allocated_bytes']} bytes, "
        f"reserved={memory['reserved_bytes']} bytes, "
        f"max_allocated={memory['max_allocated_bytes']} bytes"
    )


def load_pairs_from_selected_csv(
    selected_pairs_csv: str | Path,
    subset_size: int,
) -> list[RugdPair]:
    """Load a deterministic pair list from a Phase 3 selected_pairs.csv file."""

    selected_pairs_csv = Path(selected_pairs_csv)
    pairs: list[RugdPair] = []
    with selected_pairs_csv.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            pairs.append(
                RugdPair(
                    sequence=row["sequence"],
                    filename=row["filename"],
                    image_path=Path(row["image_path"]),
                    mask_path=Path(row["mask_path"]),
                )
            )

    if subset_size < len(pairs):
        pairs = pairs[:subset_size]
    return pairs


def select_evaluation_pairs(args) -> list[RugdPair]:
    """Select pairs, preferring the fixed Phase 3 CSV when present."""

    selected_csv = Path(args.selected_pairs_csv)
    if selected_csv.exists():
        return load_pairs_from_selected_csv(selected_csv, args.subset_size)
    return discover_image_mask_pairs(
        args.images_root,
        args.masks_root,
        subset_size=args.subset_size,
        seed=args.seed,
    )


def risk_percentages(risk_mask: np.ndarray) -> dict[str, float]:
    """Return pixel percentages for risk values 0, 1, and 2."""

    total = risk_mask.size
    return {
        f"risk_{risk_value}_pct": float((risk_mask == risk_value).sum() / total * 100)
        for risk_value in (0, 1, 2)
    }


def mean_or_empty(values: np.ndarray) -> float | str:
    """Return a mean float or an empty string when no pixels are selected."""

    if values.size == 0:
        return ""
    return float(values.mean())


def top_labels_in_mask(
    class_mask: np.ndarray,
    selected_pixels: np.ndarray,
    id2label: dict[int, str],
    limit: int = 5,
) -> str:
    """Summarize top ADE20K labels inside a boolean pixel mask."""

    if not selected_pixels.any():
        return ""

    labels, counts = np.unique(class_mask[selected_pixels], return_counts=True)
    ranked = sorted(
        ((int(label), int(count)) for label, count in zip(labels, counts)),
        key=lambda item: item[1],
        reverse=True,
    )
    total = int(selected_pixels.sum())
    parts = []
    for class_id, count in ranked[:limit]:
        label_name = id2label.get(class_id, f"class_{class_id}")
        parts.append(f"{class_id}:{label_name}={count} ({count / total * 100:.2f}%)")
    return "; ".join(parts)


def top_labels_for_print(class_mask: np.ndarray, id2label: dict[int, str], limit: int = 8) -> list[str]:
    """Return formatted top semantic labels for console diagnostics."""

    labels, counts = np.unique(class_mask, return_counts=True)
    total = class_mask.size
    ranked = sorted(
        ((int(label), int(count)) for label, count in zip(labels, counts)),
        key=lambda item: item[1],
        reverse=True,
    )
    return [
        f"{class_id}:{id2label.get(class_id, f'class_{class_id}')}={count} ({count / total * 100:.2f}%)"
        for class_id, count in ranked[:limit]
    ]


def risk_distribution_for_print(risk_mask: np.ndarray) -> str:
    """Return low/medium/high risk percentages for console diagnostics."""

    percentages = risk_percentages(risk_mask)
    return (
        f"low={percentages['risk_0_pct']:.2f}%, "
        f"medium={percentages['risk_1_pct']:.2f}%, "
        f"high={percentages['risk_2_pct']:.2f}%"
    )


def run_segformer_eval(args) -> dict:
    """Evaluate zero-shot SegFormer risk predictions on a deterministic subset."""

    start_time = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir = output_dir / "qualitative_examples"
    qualitative_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(args.settings)
    model_settings = settings.get("models", {})
    requested_model = (
        args.model_name
        or
        args.segmentation_model
        or settings.get("segmentation_model")
        or model_settings.get("segmentation_model")
        or DEFAULT_SEGFORMER_MODEL
    )
    model_name = resolve_segformer_model_name(requested_model)
    device = args.device or settings.get("device") or model_settings.get("device", "auto")

    pairs = select_evaluation_pairs(args)
    selected_rows = [
        {
            "sequence": pair.sequence,
            "filename": pair.filename,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
        }
        for pair in pairs
    ]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    print_cuda_memory("GPU memory before model loading")
    segmenter = create_ade20k_segmenter(model_name=model_name, device=device)
    print_cuda_memory("GPU memory after model loading")
    device_used = str(segmenter.device)

    aggregate_matrix = np.zeros((3, 3), dtype=np.int64)
    per_image_rows: list[dict] = []
    failures: list[dict] = []

    for index, pair in enumerate(pairs):
        print(f"[{index + 1}/{len(pairs)}] {pair.sequence}/{pair.filename}")
        try:
            image_rgb, annotation_rgb = load_pair(pair)
            gt_risk_mask = annotation_rgb_to_risk_mask(
                annotation_rgb,
                labels_path=args.labels,
                risk_mapping_path=args.risk_mapping,
            )
            prediction = segmenter.predict(image_rgb)
            pred_risk_mask, _ = ade20k_class_id_to_risk_mask(
                prediction.class_id_mask,
                prediction.id2label,
                mapping_path=args.ade20k_risk_mapping,
            )

            matrix, metrics = evaluate_risk_masks(gt_risk_mask, pred_risk_mask)
            aggregate_matrix += matrix

            if args.subset_size <= 3:
                print("  Top ADE20K labels:")
                for item in top_labels_for_print(prediction.class_id_mask, prediction.id2label):
                    print(f"    {item}")
                print(f"  Predicted risk distribution: {risk_distribution_for_print(pred_risk_mask)}")
                print(f"  High-risk recall: {metrics.high_risk_recall}")
                print(f"  Unsafe-to-safe error: {metrics.unsafe_to_safe_error_rate}")
                print(f"  Safe-to-high-risk rate: {metrics.safe_to_high_risk_rate}")

            row = {
                "sequence": pair.sequence,
                "filename": pair.filename,
                "image_path": str(pair.image_path),
                "mask_path": str(pair.mask_path),
            }
            row.update(flatten_metrics(metrics))
            per_image_rows.append(row)

            if len(per_image_rows) <= 5:
                panel_path = qualitative_dir / f"{len(per_image_rows):02d}_{pair.sequence}_{Path(pair.filename).stem}_comparison.png"
                save_segformer_comparison_panel(
                    panel_path,
                    image_rgb,
                    gt_risk_mask,
                    pred_risk_mask,
                    prediction.confidence,
                )
        except Exception as exc:  # noqa: BLE001 - experiment should continue per image.
            failures.append(
                {
                    "sequence": pair.sequence,
                    "filename": pair.filename,
                    "image_path": str(pair.image_path),
                    "mask_path": str(pair.mask_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            print(f"  FAILED: {exc}")

    summary_metrics = metrics_from_confusion_matrix(aggregate_matrix)
    elapsed_seconds = time.perf_counter() - start_time

    selected_pairs_path = output_dir / "selected_pairs.csv"
    per_image_path = output_dir / "per_image_metrics.csv"
    summary_path = output_dir / "summary_metrics.json"
    runtime_path = output_dir / "runtime_summary.json"
    matrix_npy_path = output_dir / "confusion_matrix.npy"
    matrix_csv_path = output_dir / "confusion_matrix.csv"
    failures_path = output_dir / "failures.csv"

    write_csv(selected_pairs_path, selected_rows, ["sequence", "filename", "image_path", "mask_path"])

    per_image_fields = [
        "sequence",
        "filename",
        "image_path",
        "mask_path",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "high_risk_recall",
        "unsafe_to_safe_error_rate",
        "unsafe_to_medium_error_rate",
        "safe_to_high_risk_rate",
        "total_pixels",
        "precision_low_risk",
        "precision_medium_risk",
        "precision_high_risk",
        "recall_low_risk",
        "recall_medium_risk",
        "recall_high_risk",
        "f1_low_risk",
        "f1_medium_risk",
        "f1_high_risk",
    ]
    write_csv(per_image_path, per_image_rows, per_image_fields)
    write_csv(failures_path, failures, ["sequence", "filename", "image_path", "mask_path", "error", "traceback"])

    np.save(matrix_npy_path, aggregate_matrix)
    write_confusion_matrix_csv(matrix_csv_path, aggregate_matrix)

    worst_unsafe_to_safe = sorted(
        per_image_rows,
        key=lambda row: -1 if row["unsafe_to_safe_error_rate"] is None else row["unsafe_to_safe_error_rate"],
        reverse=True,
    )[:5]

    summary = {
        "mode": args.mode,
        "model": model_name,
        "model_key": args.model_name,
        "device_used": device_used,
        "subset_size_requested": args.subset_size,
        "seed": args.seed,
        "successful_images": len(per_image_rows),
        "failed_images": len(failures),
        "total_pixels_evaluated": summary_metrics.total_pixels,
        "elapsed_seconds": elapsed_seconds,
        "runtime_per_image_seconds": elapsed_seconds / len(per_image_rows)
        if per_image_rows
        else None,
        "cuda_memory": cuda_memory_summary(),
        "aggregate_metrics": summary_metrics.to_dict(),
        "confusion_matrix": aggregate_matrix.tolist(),
        "worst_images_by_unsafe_to_safe_error": [
            {
                "sequence": row["sequence"],
                "filename": row["filename"],
                "image_path": row["image_path"],
                "unsafe_to_safe_error_rate": row["unsafe_to_safe_error_rate"],
                "high_risk_recall": row["high_risk_recall"],
                "accuracy": row["accuracy"],
            }
            for row in worst_unsafe_to_safe
        ],
        "outputs": {
            "per_image_metrics_csv": str(per_image_path),
            "summary_metrics_json": str(summary_path),
            "runtime_summary_json": str(runtime_path),
            "confusion_matrix_npy": str(matrix_npy_path),
            "confusion_matrix_csv": str(matrix_csv_path),
            "selected_pairs_csv": str(selected_pairs_path),
            "failures_csv": str(failures_path),
            "qualitative_examples_dir": str(qualitative_dir),
        },
    }

    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    runtime_summary = {
        "model": model_name,
        "model_key": args.model_name,
        "device_used": device_used,
        "successful_images": len(per_image_rows),
        "failed_images": len(failures),
        "elapsed_seconds": elapsed_seconds,
        "runtime_per_image_seconds": summary["runtime_per_image_seconds"],
        "cuda_memory": summary["cuda_memory"],
    }
    with runtime_path.open("w", encoding="utf-8") as file:
        json.dump(runtime_summary, file, indent=2)

    print()
    print("Segmentation evaluation complete")
    print(f"Device used: {device_used}")
    print(f"Successful images: {len(per_image_rows)}")
    print(f"Failed images: {len(failures)}")
    print(f"Total pixels evaluated: {summary_metrics.total_pixels}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Runtime per image seconds: {summary['runtime_per_image_seconds']}")
    peak_bytes = summary["cuda_memory"].get("max_allocated_bytes")
    if peak_bytes is not None:
        print(f"Peak GPU memory MB: {peak_bytes / (1024 * 1024):.2f}")
    print("Aggregate metrics:")
    for key, value in flatten_metrics(summary_metrics).items():
        print(f"  {key}: {value}")
    print("Confusion matrix rows=GT, cols=prediction:")
    print(aggregate_matrix)
    print("Five worst images by unsafe-to-safe error:")
    for row in worst_unsafe_to_safe:
        print(
            f"  {row['sequence']}/{row['filename']}: "
            f"unsafe_to_safe={row['unsafe_to_safe_error_rate']}, "
            f"high_risk_recall={row['high_risk_recall']}"
        )
    if failures:
        print("Failures:")
        for failure in failures:
            print(f"  {failure['sequence']}/{failure['filename']}: {failure['error']}")
    print("Output paths:")
    for path in summary["outputs"].values():
        print(f"  {path}")

    if args.model_name == "segformer_b2" and args.subset_size == 30:
        create_model_comparison_csv(
            Path("outputs/phase3_eval/summary_metrics.json"),
            summary_path,
            Path("outputs/phase4_model_comparison.csv"),
        )

    if torch.cuda.is_available():
        clear_model_caches()

    return summary


def comparison_row(model_label: str, summary: dict) -> dict:
    """Create one model-comparison row from a summary JSON object."""

    metrics = summary["aggregate_metrics"]
    return {
        "model": model_label,
        "model_id": summary["model"],
        "device_used": summary["device_used"],
        "overall_accuracy": metrics["accuracy"],
        "balanced_accuracy": metrics["balanced_accuracy"],
        "macro_f1": metrics["macro_f1"],
        "high_risk_recall": metrics["high_risk_recall"],
        "unsafe_to_safe_error_rate": metrics["unsafe_to_safe_error_rate"],
        "unsafe_to_medium_error_rate": metrics["unsafe_to_medium_error_rate"],
        "safe_to_high_risk_rate": metrics["safe_to_high_risk_rate"],
        "runtime_per_image_seconds": summary.get("runtime_per_image_seconds")
        or summary["elapsed_seconds"] / summary["successful_images"],
    }


def create_model_comparison_csv(
    b0_summary_path: Path,
    b2_summary_path: Path,
    output_path: Path,
) -> None:
    """Create a B0 vs B2 comparison CSV from summary metric JSON files."""

    if not b0_summary_path.exists() or not b2_summary_path.exists():
        return

    with b0_summary_path.open("r", encoding="utf-8") as file:
        b0_summary = json.load(file)
    with b2_summary_path.open("r", encoding="utf-8") as file:
        b2_summary = json.load(file)

    rows = [
        comparison_row("segformer_b0", b0_summary),
        comparison_row("segformer_b2", b2_summary),
    ]
    write_csv(
        output_path,
        rows,
        [
            "model",
            "model_id",
            "device_used",
            "overall_accuracy",
            "balanced_accuracy",
            "macro_f1",
            "high_risk_recall",
            "unsafe_to_safe_error_rate",
            "unsafe_to_medium_error_rate",
            "safe_to_high_risk_rate",
            "runtime_per_image_seconds",
        ],
    )
    print(f"Model comparison CSV: {output_path}")


def run_failure_analysis(args) -> list[dict]:
    """Analyze the top unsafe-to-safe failure images from Phase 3."""

    start_time = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(args.settings)
    model_settings = settings.get("models", {})
    model_name = (
        args.segmentation_model
        or settings.get("segmentation_model")
        or model_settings.get("segmentation_model")
        or DEFAULT_SEGFORMER_MODEL
    )
    device = args.device or settings.get("device") or model_settings.get("device", "auto")
    segmenter = create_ade20k_segmenter(model_name=model_name, device=device)

    rows: list[dict] = []
    for sequence, filename in FAILURE_ANALYSIS_IMAGES:
        pair = find_pair(sequence, filename, args.images_root, args.masks_root)
        print(f"Analyzing {sequence}/{filename}")

        image_rgb, annotation_rgb = load_pair(pair)
        gt_risk_mask = annotation_rgb_to_risk_mask(
            annotation_rgb,
            labels_path=args.labels,
            risk_mapping_path=args.risk_mapping,
        )
        prediction = segmenter.predict(image_rgb)
        pred_risk_mask, _ = ade20k_class_id_to_risk_mask(
            prediction.class_id_mask,
            prediction.id2label,
            mapping_path=args.ade20k_risk_mapping,
        )

        _, metrics = evaluate_risk_masks(gt_risk_mask, pred_risk_mask)
        correct_high = (gt_risk_mask == 2) & (pred_risk_mask == 2)
        unsafe_to_safe = (gt_risk_mask == 2) & (pred_risk_mask == 0)

        panel_path = output_dir / f"{sequence}_{Path(filename).stem.replace(sequence + '_', '')}_failure_panel.png"
        save_failure_analysis_panel(
            panel_path,
            image_rgb,
            gt_risk_mask,
            pred_risk_mask,
            prediction.confidence,
        )

        gt_pct = risk_percentages(gt_risk_mask)
        pred_pct = risk_percentages(pred_risk_mask)
        row = {
            "sequence": sequence,
            "filename": filename,
            "image_path": str(pair.image_path),
            "panel_path": str(panel_path),
            "unsafe_to_safe_pct": None
            if metrics.unsafe_to_safe_error_rate is None
            else metrics.unsafe_to_safe_error_rate * 100,
            "high_risk_recall_pct": None
            if metrics.high_risk_recall is None
            else metrics.high_risk_recall * 100,
            "mean_conf_correct_high": mean_or_empty(prediction.confidence[correct_high]),
            "mean_conf_unsafe_to_safe": mean_or_empty(prediction.confidence[unsafe_to_safe]),
            "top_ade20k_labels_in_unsafe_to_safe": top_labels_in_mask(
                prediction.class_id_mask,
                unsafe_to_safe,
                prediction.id2label,
            ),
        }
        row.update({f"gt_{key}": value for key, value in gt_pct.items()})
        row.update({f"pred_{key}": value for key, value in pred_pct.items()})
        rows.append(row)

        print(f"  GT risk percentages: {gt_pct}")
        print(f"  Predicted risk percentages: {pred_pct}")
        print(f"  unsafe-to-safe: {row['unsafe_to_safe_pct']:.2f}%")
        print(f"  high-risk recall: {row['high_risk_recall_pct']:.2f}%")
        print(f"  mean max softmax on correct high-risk: {row['mean_conf_correct_high']}")
        print(f"  mean max softmax on unsafe-to-safe: {row['mean_conf_unsafe_to_safe']}")
        print(f"  top ADE20K labels inside unsafe-to-safe: {row['top_ade20k_labels_in_unsafe_to_safe']}")

    summary_path = output_dir / "failure_summary.csv"
    fields = [
        "sequence",
        "filename",
        "image_path",
        "panel_path",
        "gt_risk_0_pct",
        "gt_risk_1_pct",
        "gt_risk_2_pct",
        "pred_risk_0_pct",
        "pred_risk_1_pct",
        "pred_risk_2_pct",
        "unsafe_to_safe_pct",
        "high_risk_recall_pct",
        "mean_conf_correct_high",
        "mean_conf_unsafe_to_safe",
        "top_ade20k_labels_in_unsafe_to_safe",
    ]
    write_csv(summary_path, rows, fields)

    readme_path = output_dir / "README.txt"
    readme_path.write_text(
        "Phase 3.1 safety failure analysis for the existing 30-image SegFormer evaluation.\n"
        "Error map colors:\n"
        "- dark gray: correct prediction\n"
        "- magenta: GT high risk predicted low risk, critical unsafe error\n"
        "- orange: GT high risk predicted medium risk, caution error\n"
        "- cyan: GT low risk predicted high risk, conservative false alarm\n"
        "- white: other disagreement\n\n"
        "Maximum softmax probability is a confidence proxy only, not calibrated uncertainty.\n",
        encoding="utf-8",
    )

    elapsed_seconds = time.perf_counter() - start_time
    print()
    print("Failure analysis complete")
    print(f"Device used: {segmenter.device}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print(f"Summary CSV: {summary_path}")
    print(f"README: {readme_path}")
    return rows


def run_weather_eval(args) -> dict:
    """Evaluate SegFormer-B2 robustness under deterministic synthetic weather."""

    start_time = time.perf_counter()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    qualitative_dir = output_dir / "qualitative_examples"
    figures_dir = output_dir / "figures"
    qualitative_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_name = resolve_segformer_model_name(args.model_name or "segformer_b2")
    device = args.device or "auto"
    pairs = select_evaluation_pairs(args)
    selected_rows = [
        {
            "sequence": pair.sequence,
            "filename": pair.filename,
            "image_path": str(pair.image_path),
            "mask_path": str(pair.mask_path),
        }
        for pair in pairs
    ]

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
    print_cuda_memory("GPU memory before model loading")
    segmenter = create_ade20k_segmenter(model_name=model_name, device=device)
    print_cuda_memory("GPU memory after model loading")

    aggregate_by_condition = {condition: np.zeros((3, 3), dtype=np.int64) for condition in CONDITIONS}
    confidence_values = {condition: [] for condition in CONDITIONS}
    runtime_values = {condition: [] for condition in CONDITIONS}
    per_image_rows: list[dict] = []
    failures: list[dict] = []
    difficult_rows = []

    for pair_index, pair in enumerate(pairs):
        print(f"[{pair_index + 1}/{len(pairs)}] {pair.sequence}/{pair.filename}")
        try:
            image_rgb, annotation_rgb = load_pair(pair)
            gt_risk_mask = annotation_rgb_to_risk_mask(annotation_rgb)
        except Exception as exc:  # noqa: BLE001
            failures.append(
                {
                    "condition": "all",
                    "sequence": pair.sequence,
                    "filename": pair.filename,
                    "image_path": str(pair.image_path),
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                }
            )
            continue

        for condition_index, condition in enumerate(CONDITIONS):
            try:
                condition_seed = args.seed + pair_index * 100 + condition_index
                degraded = apply_weather(
                    image_rgb,
                    condition,
                    severity=WEATHER_SEVERITY,
                    seed=condition_seed,
                )
                inference_start = time.perf_counter()
                prediction = segmenter.predict(degraded)
                pred_risk_mask, _ = ade20k_class_id_to_risk_mask(
                    prediction.class_id_mask,
                    prediction.id2label,
                )
                runtime = time.perf_counter() - inference_start
                matrix, metrics = evaluate_risk_masks(gt_risk_mask, pred_risk_mask)
                aggregate_by_condition[condition] += matrix
                confidence_values[condition].append(float(prediction.confidence.mean()))
                runtime_values[condition].append(runtime)

                row = {
                    "condition": condition,
                    "sequence": pair.sequence,
                    "filename": pair.filename,
                    "image_path": str(pair.image_path),
                    "mean_confidence_proxy": float(prediction.confidence.mean()),
                    "runtime_seconds": runtime,
                }
                row.update(flatten_metrics(metrics))
                per_image_rows.append(row)

                if pair_index < 5:
                    panel_path = qualitative_dir / f"{condition}_{pair_index + 1:02d}_{pair.sequence}_{Path(pair.filename).stem}.png"
                    save_weather_comparison_panel(
                        panel_path,
                        degraded,
                        gt_risk_mask,
                        pred_risk_mask,
                        prediction.confidence,
                        condition,
                    )

                if pair_index == 0:
                    difficult_rows.append(
                        (condition, degraded, gt_risk_mask, pred_risk_mask, prediction.confidence)
                    )
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    {
                        "condition": condition,
                        "sequence": pair.sequence,
                        "filename": pair.filename,
                        "image_path": str(pair.image_path),
                        "error": str(exc),
                        "traceback": traceback.format_exc(),
                    }
                )
                print(f"  FAILED {condition}: {exc}")

    summary_rows = []
    for condition in CONDITIONS:
        metrics = metrics_from_confusion_matrix(aggregate_by_condition[condition])
        row = {
            "condition": condition,
            "accuracy": metrics.accuracy,
            "balanced_accuracy": metrics.balanced_accuracy,
            "macro_f1": metrics.macro_f1,
            "high_risk_recall": metrics.high_risk_recall,
            "unsafe_to_safe_error_rate": metrics.unsafe_to_safe_error_rate,
            "unsafe_to_medium_error_rate": metrics.unsafe_to_medium_error_rate,
            "safe_to_high_risk_rate": metrics.safe_to_high_risk_rate,
            "mean_confidence_proxy": float(np.mean(confidence_values[condition]))
            if confidence_values[condition]
            else "",
            "runtime_per_image_seconds": float(np.mean(runtime_values[condition]))
            if runtime_values[condition]
            else "",
            "total_pixels": metrics.total_pixels,
        }
        summary_rows.append(row)

    write_csv(output_dir / "selected_pairs.csv", selected_rows, ["sequence", "filename", "image_path", "mask_path"])
    write_csv(
        output_dir / "per_image_metrics.csv",
        per_image_rows,
        [
            "condition",
            "sequence",
            "filename",
            "image_path",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "high_risk_recall",
            "unsafe_to_safe_error_rate",
            "unsafe_to_medium_error_rate",
            "safe_to_high_risk_rate",
            "mean_confidence_proxy",
            "runtime_seconds",
            "total_pixels",
            "precision_low_risk",
            "precision_medium_risk",
            "precision_high_risk",
            "recall_low_risk",
            "recall_medium_risk",
            "recall_high_risk",
            "f1_low_risk",
            "f1_medium_risk",
            "f1_high_risk",
        ],
    )
    write_csv(
        output_dir / "summary_by_condition.csv",
        summary_rows,
        [
            "condition",
            "accuracy",
            "balanced_accuracy",
            "macro_f1",
            "high_risk_recall",
            "unsafe_to_safe_error_rate",
            "unsafe_to_medium_error_rate",
            "safe_to_high_risk_rate",
            "mean_confidence_proxy",
            "runtime_per_image_seconds",
            "total_pixels",
        ],
    )
    write_csv(
        output_dir / "failures.csv",
        failures,
        ["condition", "sequence", "filename", "image_path", "error", "traceback"],
    )

    weather_config = {
        "model_name": args.model_name or "segformer_b2",
        "resolved_model": model_name,
        "conditions": CONDITIONS,
        "seed": args.seed,
        "severity": WEATHER_SEVERITY,
        "selected_pairs_csv": str(args.selected_pairs_csv),
        "gt_masks_transformed": False,
    }
    (output_dir / "weather_config.json").write_text(json.dumps(weather_config, indent=2), encoding="utf-8")
    (output_dir / "README.txt").write_text(
        "Phase 5 synthetic weather robustness evaluation for SegFormer-B2.\n"
        "Only RGB inputs are degraded; RUGD-derived ground-truth risk masks are unchanged.\n"
        "Weather is deterministic with seed 42 and moderate severity. These synthetic effects are not real sensor/weather models.\n",
        encoding="utf-8",
    )

    labels = [row["condition"] for row in summary_rows]
    save_metric_bar_chart(
        figures_dir / "condition_vs_high_risk_recall.png",
        labels,
        [float(row["high_risk_recall"]) for row in summary_rows],
        "Condition vs high-risk recall",
        "High-risk recall",
    )
    save_metric_bar_chart(
        figures_dir / "condition_vs_unsafe_to_safe_error_rate.png",
        labels,
        [float(row["unsafe_to_safe_error_rate"]) for row in summary_rows],
        "Condition vs unsafe-to-safe error rate",
        "Unsafe-to-safe error rate",
    )
    save_metric_bar_chart(
        figures_dir / "condition_vs_mean_confidence_proxy.png",
        labels,
        [float(row["mean_confidence_proxy"]) for row in summary_rows],
        "Condition vs mean confidence proxy",
        "Mean maximum softmax probability",
    )
    if difficult_rows:
        save_weather_rows_figure(figures_dir / "difficult_creek_weather_rows.png", difficult_rows)

    elapsed_seconds = time.perf_counter() - start_time
    print()
    print("Weather evaluation complete")
    print(f"Device used: {segmenter.device}")
    print(f"Successful image-condition cases: {len(per_image_rows)}")
    print(f"Failed image-condition cases: {len(failures)}")
    print(f"Elapsed seconds: {elapsed_seconds:.2f}")
    print("Metrics by condition:")
    for row in summary_rows:
        print(
            f"  {row['condition']}: acc={row['accuracy']:.4f}, "
            f"bal_acc={row['balanced_accuracy']:.4f}, macro_f1={row['macro_f1']:.4f}, "
            f"high_recall={row['high_risk_recall']:.4f}, unsafe_safe={row['unsafe_to_safe_error_rate']:.4f}, "
            f"unsafe_med={row['unsafe_to_medium_error_rate']:.4f}, safe_high={row['safe_to_high_risk_rate']:.4f}, "
            f"mean_conf={row['mean_confidence_proxy']:.4f}, runtime_img={row['runtime_per_image_seconds']:.4f}"
        )

    if torch.cuda.is_available():
        clear_model_caches()

    return {
        "summary_rows": summary_rows,
        "failures": failures,
        "elapsed_seconds": elapsed_seconds,
        "device": str(segmenter.device),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a subset experiment.")
    parser.add_argument(
        "--mode",
        choices=["segformer_eval", "failure_analysis", "weather_eval"],
        default="segformer_eval",
    )
    parser.add_argument("--subset_size", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", default="outputs/phase3_eval")
    parser.add_argument("--images_root", default=str(DEFAULT_IMAGES_ROOT))
    parser.add_argument("--masks_root", default=str(DEFAULT_MASKS_ROOT))
    parser.add_argument("--labels", default="config/rugd_labels.yaml")
    parser.add_argument("--risk_mapping", default="config/risk_mapping.yaml")
    parser.add_argument("--ade20k_risk_mapping", default="config/ade20k_risk_mapping.yaml")
    parser.add_argument("--settings", default="config/settings.yaml")
    parser.add_argument("--selected_pairs_csv", default=str(DEFAULT_SELECTED_PAIRS_CSV))
    parser.add_argument("--model_name", choices=sorted(MODEL_REGISTRY), default=None)
    parser.add_argument("--segmentation_model", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    if args.mode == "segformer_eval":
        run_segformer_eval(args)
    elif args.mode == "failure_analysis":
        run_failure_analysis(args)
    elif args.mode == "weather_eval":
        run_weather_eval(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
