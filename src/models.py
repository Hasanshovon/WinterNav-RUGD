"""Inference-only wrappers for pretrained semantic segmentation models."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


DEFAULT_SEGFORMER_MODEL = "nvidia/segformer-b0-finetuned-ade-512-512"
MODEL_REGISTRY = {
    "segformer_b0": "nvidia/segformer-b0-finetuned-ade-512-512",
    "segformer_b2": "nvidia/segformer-b2-finetuned-ade-512-512",
    "mask2former_swin_small": "facebook/mask2former-swin-small-ade-semantic",
    "upernet_convnext_tiny": "openmmlab/upernet-convnext-tiny",
}
MODEL_ARCHITECTURES = {
    "segformer_b0": "segformer",
    "segformer_b2": "segformer",
    "mask2former_swin_small": "mask2former",
    "upernet_convnext_tiny": "upernet",
}
SEGFORMER_MODEL_REGISTRY = MODEL_REGISTRY


@dataclass(frozen=True)
class SegFormerPrediction:
    """ADE20K semantic prediction resized to the original image size."""

    class_id_mask: np.ndarray
    confidence: np.ndarray
    id2label: dict[int, str]
    device: str
    model_name: str


class SegFormerADE20K:
    """Small wrapper for zero-shot ADE20K SegFormer inference.

    The returned class IDs and labels are ADE20K predictions from a model
    pretrained on ADE20K. They are not RUGD semantic labels.
    """

    _processor = None
    _model = None
    _model_name = None
    _device = None

    def __init__(self, model_name: str = DEFAULT_SEGFORMER_MODEL, device: str = "auto"):
        self.model_name = resolve_segmentation_model_name(model_name)
        self.device = self._resolve_device(device)
        self._load_once()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    def _load_once(self) -> None:
        if (
            SegFormerADE20K._model is not None
            and SegFormerADE20K._processor is not None
            and SegFormerADE20K._model_name == self.model_name
            and SegFormerADE20K._device == str(self.device)
        ):
            return

        from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = AutoModelForSemanticSegmentation.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        SegFormerADE20K._processor = processor
        SegFormerADE20K._model = model
        SegFormerADE20K._model_name = self.model_name
        SegFormerADE20K._device = str(self.device)

    def predict(self, image_rgb: np.ndarray) -> SegFormerPrediction:
        """Predict ADE20K classes and max softmax probability for an RGB image."""

        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (height, width, 3)")

        processor = SegFormerADE20K._processor
        model = SegFormerADE20K._model
        if processor is None or model is None:
            raise RuntimeError("SegFormer model was not loaded")

        original_height, original_width = image_rgb.shape[:2]
        image = Image.fromarray(image_rgb)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits

            resized_logits = F.interpolate(
                logits,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = resized_logits.softmax(dim=1)
            confidence, class_ids = probabilities.max(dim=1)
            _ = class_ids

            discrete = logits.argmax(dim=1, keepdim=True).float()
            resized_discrete = F.interpolate(
                discrete,
                size=(original_height, original_width),
                mode="nearest",
            ).squeeze(1)

        id2label = {
            int(class_id): str(label)
            for class_id, label in model.config.id2label.items()
        }

        return SegFormerPrediction(
            class_id_mask=resized_discrete.squeeze(0).cpu().numpy().astype(np.int16),
            confidence=confidence.squeeze(0).cpu().numpy().astype(np.float32),
            id2label=id2label,
            device=str(self.device),
            model_name=self.model_name,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Release cached model references and clear CUDA cache when available."""

        cls._processor = None
        cls._model = None
        cls._model_name = None
        cls._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class Mask2FormerADE20K:
    """Wrapper for ADE20K Mask2Former semantic segmentation.

    The semantic class-ID mask is produced with the official Hugging Face image
    processor's semantic segmentation post-processing.
    """

    _processor = None
    _model = None
    _model_name = None
    _device = None

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = resolve_segmentation_model_name(model_name)
        self.device = SegFormerADE20K._resolve_device(device)
        self._load_once()

    def _load_once(self) -> None:
        if (
            Mask2FormerADE20K._model is not None
            and Mask2FormerADE20K._processor is not None
            and Mask2FormerADE20K._model_name == self.model_name
            and Mask2FormerADE20K._device == str(self.device)
        ):
            return

        from transformers import AutoImageProcessor, Mask2FormerForUniversalSegmentation

        processor = AutoImageProcessor.from_pretrained(self.model_name)
        model = Mask2FormerForUniversalSegmentation.from_pretrained(self.model_name)
        model.to(self.device)
        model.eval()

        Mask2FormerADE20K._processor = processor
        Mask2FormerADE20K._model = model
        Mask2FormerADE20K._model_name = self.model_name
        Mask2FormerADE20K._device = str(self.device)

    def predict(self, image_rgb: np.ndarray) -> SegFormerPrediction:
        """Predict ADE20K classes and confidence proxy for an RGB image."""

        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (height, width, 3)")

        processor = Mask2FormerADE20K._processor
        model = Mask2FormerADE20K._model
        if processor is None or model is None:
            raise RuntimeError("Mask2Former model was not loaded")

        original_height, original_width = image_rgb.shape[:2]
        image = Image.fromarray(image_rgb)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            processed = processor.post_process_semantic_segmentation(
                outputs,
                target_sizes=[(original_height, original_width)],
            )[0]

            class_probabilities = outputs.class_queries_logits.softmax(dim=-1)[..., :-1]
            mask_probabilities = outputs.masks_queries_logits.sigmoid()
            semantic_scores = torch.einsum(
                "bqc,bqhw->bchw",
                class_probabilities,
                mask_probabilities,
            )
            resized_scores = F.interpolate(
                semantic_scores,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            confidence = resized_scores.softmax(dim=1).max(dim=1).values

        id2label = {
            int(class_id): str(label)
            for class_id, label in model.config.id2label.items()
        }

        return SegFormerPrediction(
            class_id_mask=processed.cpu().numpy().astype(np.int16),
            confidence=confidence.squeeze(0).cpu().numpy().astype(np.float32),
            id2label=id2label,
            device=str(self.device),
            model_name=self.model_name,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Release cached model references and clear CUDA cache when available."""

        cls._processor = None
        cls._model = None
        cls._model_name = None
        cls._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def strict_load_pretrained_semantic_model(
    model_id: str,
    processor_class,
    model_class,
) -> tuple[object, object]:
    """Load a model with strict reporting and reject newly initialized trainable params."""

    print(f"Strict checkpoint validation for model ID: {model_id}")
    print("ignore_mismatched_sizes=True used: False")
    processor = processor_class.from_pretrained(model_id)
    model, loading_info = model_class.from_pretrained(
        model_id,
        output_loading_info=True,
    )

    missing_keys = list(loading_info.get("missing_keys", []))
    unexpected_keys = list(loading_info.get("unexpected_keys", []))
    mismatched_keys = list(loading_info.get("mismatched_keys", []))
    parameter_map = dict(model.named_parameters())
    newly_initialized_trainable = [
        key
        for key in missing_keys
        if key in parameter_map and parameter_map[key].requires_grad
    ]

    print(f"model_class: {model.__class__.__name__}")
    print(f"processor_class: {processor.__class__.__name__}")
    print(f"config_architecture: {getattr(model.config, 'architectures', None)}")
    print(f"missing_keys_count: {len(missing_keys)}")
    for key in missing_keys:
        print(f"  MISSING {key}")
    print(f"unexpected_keys_count: {len(unexpected_keys)}")
    for key in unexpected_keys:
        print(f"  UNEXPECTED {key}")
    print(f"mismatched_keys_count: {len(mismatched_keys)}")
    for key in mismatched_keys:
        print(f"  MISMATCHED {key}")
    print(f"newly_initialized_trainable_count: {len(newly_initialized_trainable)}")
    for key in newly_initialized_trainable:
        print(f"  NEW_TRAINABLE {key}")

    if newly_initialized_trainable:
        joined = ", ".join(newly_initialized_trainable)
        raise RuntimeError(
            "Strict checkpoint validation failed: newly initialized trainable "
            f"parameters found: {joined}"
        )

    return processor, model


class UperNetADE20K:
    """Wrapper for UPerNet ADE20K semantic segmentation."""

    _processor = None
    _model = None
    _model_name = None
    _device = None

    def __init__(self, model_name: str, device: str = "auto"):
        self.model_name = resolve_segmentation_model_name(model_name)
        self.device = SegFormerADE20K._resolve_device(device)
        self._load_once()

    def _load_once(self) -> None:
        if (
            UperNetADE20K._model is not None
            and UperNetADE20K._processor is not None
            and UperNetADE20K._model_name == self.model_name
            and UperNetADE20K._device == str(self.device)
        ):
            return

        from transformers import AutoImageProcessor, UperNetForSemanticSegmentation

        processor, model = strict_load_pretrained_semantic_model(
            self.model_name,
            AutoImageProcessor,
            UperNetForSemanticSegmentation,
        )
        model.to(self.device)
        model.eval()

        UperNetADE20K._processor = processor
        UperNetADE20K._model = model
        UperNetADE20K._model_name = self.model_name
        UperNetADE20K._device = str(self.device)

    def predict(self, image_rgb: np.ndarray) -> SegFormerPrediction:
        """Predict ADE20K classes and max softmax probability for an RGB image."""

        image_rgb = np.asarray(image_rgb, dtype=np.uint8)
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (height, width, 3)")

        processor = UperNetADE20K._processor
        model = UperNetADE20K._model
        if processor is None or model is None:
            raise RuntimeError("UPerNet model was not loaded")

        original_height, original_width = image_rgb.shape[:2]
        image = Image.fromarray(image_rgb)
        inputs = processor(images=image, return_tensors="pt")
        inputs = {key: value.to(self.device) for key, value in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs)
            resized_logits = F.interpolate(
                outputs.logits,
                size=(original_height, original_width),
                mode="bilinear",
                align_corners=False,
            )
            probabilities = resized_logits.softmax(dim=1)
            confidence, class_ids = probabilities.max(dim=1)

        id2label = {
            int(class_id): str(label)
            for class_id, label in model.config.id2label.items()
        }

        return SegFormerPrediction(
            class_id_mask=class_ids.squeeze(0).cpu().numpy().astype(np.int16),
            confidence=confidence.squeeze(0).cpu().numpy().astype(np.float32),
            id2label=id2label,
            device=str(self.device),
            model_name=self.model_name,
        )

    @classmethod
    def clear_cache(cls) -> None:
        """Release cached model references and clear CUDA cache when available."""

        cls._processor = None
        cls._model = None
        cls._model_name = None
        cls._device = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def resize_discrete_class_mask_nearest(
    class_mask: np.ndarray,
    size: tuple[int, int],
) -> np.ndarray:
    """Resize a discrete class-ID mask with nearest-neighbor interpolation."""

    tensor = torch.as_tensor(class_mask, dtype=torch.float32)[None, None, ...]
    resized = F.interpolate(tensor, size=size, mode="nearest")
    return resized.squeeze(0).squeeze(0).numpy().astype(class_mask.dtype)


def confidence_map_in_unit_interval(confidence: np.ndarray) -> bool:
    """Return true when all confidence values are finite and in [0, 1]."""

    confidence = np.asarray(confidence)
    return bool(np.isfinite(confidence).all() and np.all((0.0 <= confidence) & (confidence <= 1.0)))


def resolve_segformer_model_name(model_name: str) -> str:
    """Resolve a registry key or Hugging Face model ID to a model ID."""

    return resolve_segmentation_model_name(model_name)


def resolve_segmentation_model_name(model_name: str) -> str:
    """Resolve a registered model key or Hugging Face model ID."""

    if model_name in MODEL_REGISTRY:
        return MODEL_REGISTRY[model_name]
    if "/" in model_name:
        return model_name
    raise ValueError(
        f"Unknown model name '{model_name}'. "
        f"Known models: {', '.join(sorted(MODEL_REGISTRY))}"
    )


def model_architecture(model_name: str) -> str:
    """Return the architecture type for a registry key or model ID."""

    if model_name in MODEL_ARCHITECTURES:
        return MODEL_ARCHITECTURES[model_name]

    resolved = resolve_segmentation_model_name(model_name)
    if "mask2former" in resolved:
        return "mask2former"
    if "upernet" in resolved:
        return "upernet"
    if "segformer" in resolved:
        return "segformer"
    raise ValueError(f"Cannot infer architecture for model '{model_name}'")


def create_ade20k_segmenter(model_name: str, device: str = "auto"):
    """Create a supported ADE20K semantic segmentation wrapper."""

    architecture = model_architecture(model_name)
    resolved_name = resolve_segmentation_model_name(model_name)
    if architecture == "segformer":
        return SegFormerADE20K(model_name=resolved_name, device=device)
    if architecture == "mask2former":
        return Mask2FormerADE20K(model_name=resolved_name, device=device)
    if architecture == "upernet":
        return UperNetADE20K(model_name=resolved_name, device=device)
    raise ValueError(f"Unsupported model architecture: {architecture}")


def clear_model_caches() -> None:
    """Clear cached segmentation model references and CUDA cache."""

    SegFormerADE20K.clear_cache()
    Mask2FormerADE20K.clear_cache()
    UperNetADE20K.clear_cache()


def prediction_matches_original_size(
    class_id_mask: np.ndarray,
    confidence: np.ndarray,
    original_shape: tuple[int, int] | tuple[int, int, int],
) -> bool:
    """Return true when prediction arrays match the original image height/width."""

    expected_shape = original_shape[:2]
    return class_id_mask.shape == expected_shape and confidence.shape == expected_shape
