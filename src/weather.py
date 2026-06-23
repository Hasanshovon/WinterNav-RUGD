"""Deterministic synthetic weather and visibility degradations."""

from __future__ import annotations

import cv2
import numpy as np


CONDITIONS = ["normal", "low_light", "gaussian_blur", "fog", "synthetic_snow"]


def _validate_image(image_rgb: np.ndarray) -> np.ndarray:
    image_rgb = np.asarray(image_rgb)
    if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
        raise ValueError("image_rgb must have shape (height, width, 3)")
    if image_rgb.dtype != np.uint8:
        raise ValueError("image_rgb must have dtype uint8")
    return image_rgb


def _clip_uint8(image: np.ndarray) -> np.ndarray:
    return np.clip(image, 0, 255).astype(np.uint8)


def low_light(image_rgb: np.ndarray, severity: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Darken an RGB image with a mild contrast reduction."""

    _ = seed
    image_rgb = _validate_image(image_rgb)
    if severity <= 0:
        return image_rgb.copy()
    severity = float(np.clip(severity, 0.0, 1.0))
    factor = 1.0 - 0.55 * severity
    bias = -18.0 * severity
    return _clip_uint8(image_rgb.astype(np.float32) * factor + bias)


def gaussian_blur(image_rgb: np.ndarray, severity: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Apply deterministic Gaussian blur."""

    _ = seed
    image_rgb = _validate_image(image_rgb)
    if severity <= 0:
        return image_rgb.copy()
    severity = float(np.clip(severity, 0.0, 1.0))
    kernel_size = int(1 + 2 * round(2 + 4 * severity))
    return cv2.GaussianBlur(image_rgb, (kernel_size, kernel_size), sigmaX=1.0 + 2.0 * severity)


def fog(image_rgb: np.ndarray, severity: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Blend image with white haze and a smooth deterministic fog field."""

    image_rgb = _validate_image(image_rgb)
    if severity <= 0:
        return image_rgb.copy()
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    height, width = image_rgb.shape[:2]
    coarse = rng.uniform(0.4, 1.0, size=(max(2, height // 80), max(2, width // 80))).astype(np.float32)
    fog_field = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)
    fog_field = cv2.GaussianBlur(fog_field, (0, 0), sigmaX=35)
    fog_field = np.clip(fog_field, 0.0, 1.0)[..., None]
    alpha = (0.20 + 0.45 * severity) * fog_field
    result = image_rgb.astype(np.float32) * (1.0 - alpha) + 255.0 * alpha
    return _clip_uint8(result)


def synthetic_snow(image_rgb: np.ndarray, severity: float = 0.5, seed: int | None = None) -> np.ndarray:
    """Overlay deterministic white snow flecks and mild bright haze."""

    image_rgb = _validate_image(image_rgb)
    if severity <= 0:
        return image_rgb.copy()
    severity = float(np.clip(severity, 0.0, 1.0))
    rng = np.random.default_rng(seed)
    result = image_rgb.astype(np.float32)
    haze = 0.10 + 0.18 * severity
    result = result * (1.0 - haze) + 255.0 * haze

    height, width = image_rgb.shape[:2]
    snow_probability = 0.006 + 0.035 * severity
    flakes = rng.random((height, width)) < snow_probability
    flakes = cv2.dilate(flakes.astype(np.uint8), np.ones((2, 2), dtype=np.uint8), iterations=1).astype(bool)
    result[flakes] = 255.0
    return _clip_uint8(result)


def apply_weather(
    image_rgb: np.ndarray,
    condition: str,
    severity: float = 0.5,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a named deterministic weather condition to an RGB image."""

    if condition == "normal":
        return _validate_image(image_rgb).copy()
    if condition == "low_light":
        return low_light(image_rgb, severity=severity, seed=seed)
    if condition == "gaussian_blur":
        return gaussian_blur(image_rgb, severity=severity, seed=seed)
    if condition == "fog":
        return fog(image_rgb, severity=severity, seed=seed)
    if condition == "synthetic_snow":
        return synthetic_snow(image_rgb, severity=severity, seed=seed)
    raise ValueError(f"Unknown weather condition: {condition}")
