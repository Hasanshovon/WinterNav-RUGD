"""Visualization utilities for Phase 1 traversability outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from src.traversability import risk_mask_to_color


def make_risk_overlay(
    image_rgb: np.ndarray,
    risk_mask: np.ndarray,
    alpha: float = 0.45,
) -> np.ndarray:
    """Blend a green/yellow/red risk map over an RGB image."""

    if not 0 <= alpha <= 1:
        raise ValueError("alpha must be between 0 and 1")

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    risk_color = risk_mask_to_color(risk_mask)

    if image_rgb.shape != risk_color.shape:
        raise ValueError(
            f"Image and risk map must have the same shape. "
            f"Got image={image_rgb.shape}, risk={risk_color.shape}"
        )

    overlay = (1 - alpha) * image_rgb.astype(np.float32) + alpha * risk_color.astype(np.float32)
    return np.clip(overlay, 0, 255).astype(np.uint8)


def make_four_panel(
    image_rgb: np.ndarray,
    semantic_mask_rgb: np.ndarray,
    risk_mask: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Create a four-panel visualization for one RUGD example."""

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    semantic_mask_rgb = np.asarray(semantic_mask_rgb, dtype=np.uint8)
    risk_color = risk_mask_to_color(risk_mask)
    overlay = make_risk_overlay(image_rgb, risk_mask, alpha=alpha)

    panels = [
        ("RGB image", image_rgb),
        ("RUGD annotation", semantic_mask_rgb),
        ("GT risk map", risk_color),
        ("Risk overlay", overlay),
    ]

    height, width = image_rgb.shape[:2]
    title_height = 28
    gap = 12
    canvas = Image.new(
        "RGB",
        (4 * width + 3 * gap, height + title_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (title, array) in enumerate(panels):
        x = index * (width + gap)
        draw.text((x + 4, 8), title, fill=(0, 0, 0), font=font)
        canvas.paste(Image.fromarray(array), (x, title_height))

    return canvas


def save_four_panel(
    output_path: str | Path,
    image_rgb: np.ndarray,
    semantic_mask_rgb: np.ndarray,
    risk_mask: np.ndarray,
    alpha: float = 0.45,
) -> Path:
    """Save a four-panel visualization and return its path."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel = make_four_panel(image_rgb, semantic_mask_rgb, risk_mask, alpha=alpha)
    panel.save(output_path)
    return output_path


def confidence_to_heatmap(confidence: np.ndarray) -> np.ndarray:
    """Convert a [0, 1] confidence map to a simple blue-to-red RGB heatmap."""

    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.ndim != 2:
        raise ValueError("confidence must have shape (height, width)")
    confidence = np.clip(confidence, 0.0, 1.0)

    red = (confidence * 255).astype(np.uint8)
    green = ((1.0 - np.abs(confidence - 0.5) * 2.0) * 255).astype(np.uint8)
    blue = ((1.0 - confidence) * 255).astype(np.uint8)
    return np.stack([red, green, blue], axis=2)


def save_confidence_heatmap_with_colorbar(
    output_path: str | Path,
    confidence: np.ndarray,
) -> Path:
    """Save a confidence heatmap with an explicit 0.0 to 1.0 colorbar."""

    import matplotlib.pyplot as plt

    confidence = np.asarray(confidence, dtype=np.float32)
    if confidence.ndim != 2:
        raise ValueError("confidence must have shape (height, width)")
    confidence = np.clip(confidence, 0.0, 1.0)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    image = ax.imshow(confidence, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.axis("off")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Maximum softmax probability")
    colorbar.set_ticks([0.0, 1.0])
    colorbar.set_ticklabels(["0.0 low confidence", "1.0 high confidence"])
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    return output_path


def make_segformer_comparison_panel(
    image_rgb: np.ndarray,
    gt_risk_mask: np.ndarray,
    predicted_risk_mask: np.ndarray,
    confidence: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Create RGB | GT risk | predicted risk | confidence | predicted overlay."""

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    gt_color = risk_mask_to_color(gt_risk_mask)
    predicted_color = risk_mask_to_color(predicted_risk_mask)
    confidence_heatmap = confidence_to_heatmap(confidence)
    predicted_overlay = make_risk_overlay(image_rgb, predicted_risk_mask, alpha=alpha)

    panels = [
        ("RGB image", image_rgb),
        ("RUGD GT risk", gt_color),
        ("SegFormer risk", predicted_color),
        ("Max softmax", confidence_heatmap),
        ("Pred overlay", predicted_overlay),
    ]

    height, width = image_rgb.shape[:2]
    title_height = 28
    gap = 12
    canvas = Image.new(
        "RGB",
        (5 * width + 4 * gap, height + title_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (title, array) in enumerate(panels):
        x = index * (width + gap)
        draw.text((x + 4, 8), title, fill=(0, 0, 0), font=font)
        canvas.paste(Image.fromarray(array), (x, title_height))

    return canvas


def save_segformer_comparison_panel(
    output_path: str | Path,
    image_rgb: np.ndarray,
    gt_risk_mask: np.ndarray,
    predicted_risk_mask: np.ndarray,
    confidence: np.ndarray,
    alpha: float = 0.45,
) -> Path:
    """Save a five-panel SegFormer comparison figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel = make_segformer_comparison_panel(
        image_rgb,
        gt_risk_mask,
        predicted_risk_mask,
        confidence,
        alpha=alpha,
    )
    panel.save(output_path)
    return output_path


def safety_error_map_to_color(gt_risk_mask: np.ndarray, pred_risk_mask: np.ndarray) -> np.ndarray:
    """Color safety error categories for failure analysis.

    Colors:
    - correct prediction: dark gray
    - GT high risk predicted low risk: magenta
    - GT high risk predicted medium risk: orange
    - GT low risk predicted high risk: cyan
    - other disagreement: white
    """

    gt_risk_mask = np.asarray(gt_risk_mask)
    pred_risk_mask = np.asarray(pred_risk_mask)
    if gt_risk_mask.shape != pred_risk_mask.shape:
        raise ValueError(
            f"GT and predicted masks must have the same shape. "
            f"Got gt={gt_risk_mask.shape}, pred={pred_risk_mask.shape}"
        )

    colors = np.zeros((*gt_risk_mask.shape, 3), dtype=np.uint8)
    colors[:] = [255, 255, 255]
    colors[gt_risk_mask == pred_risk_mask] = [55, 55, 55]
    colors[(gt_risk_mask == 2) & (pred_risk_mask == 0)] = [255, 0, 255]
    colors[(gt_risk_mask == 2) & (pred_risk_mask == 1)] = [255, 165, 0]
    colors[(gt_risk_mask == 0) & (pred_risk_mask == 2)] = [0, 255, 255]
    return colors


def make_failure_analysis_panel(
    image_rgb: np.ndarray,
    gt_risk_mask: np.ndarray,
    predicted_risk_mask: np.ndarray,
    confidence: np.ndarray,
    alpha: float = 0.45,
) -> Image.Image:
    """Create RGB | GT risk | predicted risk | confidence | error map | overlay."""

    image_rgb = np.asarray(image_rgb, dtype=np.uint8)
    panels = [
        ("RGB", image_rgb),
        ("GT risk", risk_mask_to_color(gt_risk_mask)),
        ("Pred risk", risk_mask_to_color(predicted_risk_mask)),
        ("Confidence proxy", confidence_to_heatmap(confidence)),
        ("Safety errors", safety_error_map_to_color(gt_risk_mask, predicted_risk_mask)),
        ("Pred overlay", make_risk_overlay(image_rgb, predicted_risk_mask, alpha=alpha)),
    ]

    height, width = image_rgb.shape[:2]
    title_height = 28
    gap = 12
    canvas = Image.new(
        "RGB",
        (6 * width + 5 * gap, height + title_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for index, (title, array) in enumerate(panels):
        x = index * (width + gap)
        draw.text((x + 4, 8), title, fill=(0, 0, 0), font=font)
        canvas.paste(Image.fromarray(array), (x, title_height))

    return canvas


def save_failure_analysis_panel(
    output_path: str | Path,
    image_rgb: np.ndarray,
    gt_risk_mask: np.ndarray,
    predicted_risk_mask: np.ndarray,
    confidence: np.ndarray,
    alpha: float = 0.45,
) -> Path:
    """Save a six-panel safety failure analysis figure."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panel = make_failure_analysis_panel(
        image_rgb,
        gt_risk_mask,
        predicted_risk_mask,
        confidence,
        alpha=alpha,
    )
    panel.save(output_path)
    return output_path


def save_weather_comparison_panel(
    output_path: str | Path,
    image_rgb: np.ndarray,
    gt_risk_mask: np.ndarray,
    predicted_risk_mask: np.ndarray,
    confidence: np.ndarray,
    condition: str,
) -> Path:
    """Save RGB | GT risk | predicted risk | confidence for one weather condition."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    panels = [
        (f"RGB {condition}", np.asarray(image_rgb, dtype=np.uint8)),
        ("GT risk", risk_mask_to_color(gt_risk_mask)),
        ("Pred risk", risk_mask_to_color(predicted_risk_mask)),
        ("Confidence", confidence_to_heatmap(confidence)),
    ]
    height, width = image_rgb.shape[:2]
    title_height = 28
    gap = 12
    canvas = Image.new("RGB", (4 * width + 3 * gap, height + title_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    for index, (title, array) in enumerate(panels):
        x = index * (width + gap)
        draw.text((x + 4, 8), title, fill=(0, 0, 0), font=font)
        canvas.paste(Image.fromarray(array), (x, title_height))
    canvas.save(output_path)
    return output_path


def save_weather_rows_figure(
    output_path: str | Path,
    rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
) -> Path:
    """Save a five-row condition figure: RGB | GT risk | predicted risk | confidence."""

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("rows must not be empty")

    height, width = rows[0][1].shape[:2]
    title_height = 28
    row_label_width = 110
    gap = 10
    row_height = height + title_height
    canvas = Image.new(
        "RGB",
        (row_label_width + 4 * width + 3 * gap, len(rows) * row_height),
        color=(255, 255, 255),
    )
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    headers = ["RGB", "GT risk", "Pred risk", "Confidence"]

    for row_index, (condition, image_rgb, gt_risk_mask, pred_risk_mask, confidence) in enumerate(rows):
        y = row_index * row_height
        draw.text((8, y + title_height + 8), condition, fill=(0, 0, 0), font=font)
        arrays = [
            image_rgb,
            risk_mask_to_color(gt_risk_mask),
            risk_mask_to_color(pred_risk_mask),
            confidence_to_heatmap(confidence),
        ]
        for col_index, (header, array) in enumerate(zip(headers, arrays)):
            x = row_label_width + col_index * (width + gap)
            draw.text((x + 4, y + 8), header, fill=(0, 0, 0), font=font)
            canvas.paste(Image.fromarray(np.asarray(array, dtype=np.uint8)), (x, y + title_height))

    canvas.save(output_path)
    return output_path


def save_metric_bar_chart(
    output_path: str | Path,
    labels: list[str],
    values: list[float],
    title: str,
    ylabel: str,
) -> Path:
    """Save a simple condition-level metric bar chart."""

    import matplotlib.pyplot as plt

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    ax.bar(labels, values, color="#4c78a8")
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)
    return output_path
