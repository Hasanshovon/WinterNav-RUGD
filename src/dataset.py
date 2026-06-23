"""RUGD RGB image and annotation-mask loading utilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class RugdPair:
    """One matched RUGD RGB image and RGB annotation mask."""

    sequence: str
    filename: str
    image_path: Path
    mask_path: Path


DEFAULT_IMAGES_ROOT = Path(
    "data/raw/rugd/RUGD_frames-with-annotations/RUGD_frames-with-annotations"
)
DEFAULT_MASKS_ROOT = Path("data/raw/rugd/RUGD_annotations/RUGD_annotations")


def load_rgb_png(path: str | Path) -> np.ndarray:
    """Load a PNG image as an RGB uint8 array."""

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def validate_pair_dimensions(image: np.ndarray, mask: np.ndarray, pair: RugdPair) -> None:
    """Raise a clear error if an image and mask do not have the same size."""

    if image.shape[:2] != mask.shape[:2]:
        raise ValueError(
            "Image and mask dimensions do not match for "
            f"{pair.sequence}/{pair.filename}: "
            f"image={image.shape[:2]}, mask={mask.shape[:2]}"
        )


def load_pair(pair: RugdPair) -> tuple[np.ndarray, np.ndarray]:
    """Load a matched RUGD RGB image and RGB annotation mask."""

    image = load_rgb_png(pair.image_path)
    mask = load_rgb_png(pair.mask_path)
    validate_pair_dimensions(image, mask, pair)
    return image, mask


def discover_image_mask_pairs(
    images_root: str | Path = DEFAULT_IMAGES_ROOT,
    masks_root: str | Path = DEFAULT_MASKS_ROOT,
    subset_size: int | None = None,
    seed: int = 42,
    strict: bool = False,
) -> list[RugdPair]:
    """Discover valid image-mask pairs recursively.

    When ``strict`` is true, missing masks raise ``FileNotFoundError``.
    When false, images without masks are skipped so partially annotated RUGD
    downloads can still be inspected.
    """

    images_root = Path(images_root)
    masks_root = Path(masks_root)

    if not images_root.exists():
        raise FileNotFoundError(f"Images root not found: {images_root}")
    if not masks_root.exists():
        raise FileNotFoundError(f"Masks root not found: {masks_root}")

    pairs: list[RugdPair] = []
    missing_masks: list[Path] = []

    for image_path in sorted(images_root.rglob("*.png")):
        relative_path = image_path.relative_to(images_root)
        mask_path = masks_root / relative_path
        if not mask_path.exists():
            missing_masks.append(relative_path)
            continue

        if len(relative_path.parts) < 2:
            sequence = image_path.parent.name
        else:
            sequence = relative_path.parts[0]

        pairs.append(
            RugdPair(
                sequence=sequence,
                filename=image_path.name,
                image_path=image_path,
                mask_path=mask_path,
            )
        )

    if strict and missing_masks:
        examples = ", ".join(path.as_posix() for path in missing_masks[:5])
        raise FileNotFoundError(
            f"Found {len(missing_masks)} image files without matching masks. "
            f"Examples: {examples}"
        )

    if not pairs:
        raise FileNotFoundError(
            f"No valid PNG image-mask pairs found under {images_root} and {masks_root}"
        )

    if subset_size is not None:
        if subset_size < 0:
            raise ValueError("subset_size must be non-negative")
        if subset_size < len(pairs):
            rng = random.Random(seed)
            pairs = sorted(rng.sample(pairs, subset_size), key=lambda pair: str(pair.image_path))

    return pairs


def find_pair(
    sequence: str,
    filename: str,
    images_root: str | Path = DEFAULT_IMAGES_ROOT,
    masks_root: str | Path = DEFAULT_MASKS_ROOT,
) -> RugdPair:
    """Find one RUGD image-mask pair by sequence and filename."""

    images_root = Path(images_root)
    masks_root = Path(masks_root)
    image_path = images_root / sequence / filename
    mask_path = masks_root / sequence / filename

    if not image_path.exists():
        raise FileNotFoundError(f"RUGD RGB image not found: {image_path}")
    if not mask_path.exists():
        raise FileNotFoundError(f"Matching RUGD annotation mask not found: {mask_path}")

    return RugdPair(
        sequence=sequence,
        filename=filename,
        image_path=image_path,
        mask_path=mask_path,
    )
