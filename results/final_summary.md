# WinterNav-RUGD Final Summary

## Project Objective

WinterNav-RUGD builds confidence-aware traversability maps for small autonomous ground vehicles using the RUGD off-road robot dataset. The prototype converts official RUGD semantic RGB masks into conservative low, medium, and high traversability risk labels, then evaluates zero-shot pretrained ADE20K semantic segmentation models against that risk ground truth.

The project intentionally avoids model training. The focus is reproducible dataset handling, conservative safety metrics, model comparison, failure analysis, and synthetic weather robustness.

## Dataset And Risk Policy

RUGD RGB images are loaded from:

```text
data/raw/rugd/RUGD_frames-with-annotations/RUGD_frames-with-annotations
```

Official RGB semantic masks are loaded from:

```text
data/raw/rugd/RUGD_annotations/RUGD_annotations
```

The official colormap is:

```text
data/raw/rugd/RUGD_annotations/RUGD_annotations/RUGD_annotation-colormap.txt
```

RUGD mask colors were verified against the official colormap. Ground-truth traversability risk is generated from verified RUGD labels using a conservative small-AGV policy:

| Risk | Meaning | Example RUGD classes |
|---:|---|---|
| 0 | low risk / likely traversable | dirt, asphalt, gravel, concrete |
| 1 | medium risk / uncertain terrain | sand, grass, mulch, bush |
| 2 | high risk / obstacle or non-traversable | rock-bed, rock, tree, log, water, vehicle, person, sky |

This policy is robot-specific and is not an official RUGD label.

## Selected Model Rationale

The main reported models are ADE20K-pretrained SegFormer variants:

| Model key | Hugging Face model |
|---|---|
| segformer_b0 | nvidia/segformer-b0-finetuned-ade-512-512 |
| segformer_b2 | nvidia/segformer-b2-finetuned-ade-512-512 |

These models provide a zero-shot domain-transfer baseline from ADE20K semantic segmentation to RUGD traversability risk. They do not predict RUGD labels directly. ADE20K labels are mapped to risk using a fixed keyword-based ADE20K-to-risk policy.

UPerNet ConvNeXt-Tiny was validated as a clean-loading smoke-test-only candidate. Mask2Former was excluded from the main benchmark because strict checkpoint validation found newly initialized trainable parameters.

## Normal-Condition Results

Fixed subset: 30 RUGD image-mask pairs from `outputs/phase3_eval/selected_pairs.csv`.

| Model | Device | Accuracy | Balanced Accuracy | Macro F1 | High-Risk Recall | Unsafe-to-Safe | Runtime/Image |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | CUDA | 0.7174 | 0.6983 | 0.6130 | 0.9196 | 0.0665 | 1.0238 s |
| SegFormer-B2 | CUDA | 0.7455 | 0.7620 | 0.6467 | 0.9446 | 0.0487 | 0.9210 s |

SegFormer-B2 improved balanced accuracy, macro F1, high-risk recall, and unsafe-to-safe error on this subset.

## Weather Robustness

SegFormer-B2 was evaluated with deterministic moderate-severity RGB corruptions. Ground-truth risk masks were not transformed.

| Condition | Accuracy | Balanced Accuracy | Macro F1 | High-Risk Recall | Unsafe-to-Safe | Unsafe-to-Medium | Safe-to-High | Mean Confidence |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| normal | 0.7455 | 0.7620 | 0.6467 | 0.9446 | 0.0487 | 0.0067 | 0.0200 | 0.9195 |
| low_light | 0.7402 | 0.7566 | 0.6399 | 0.9437 | 0.0485 | 0.0078 | 0.0200 | 0.9213 |
| gaussian_blur | 0.7198 | 0.7430 | 0.6176 | 0.9329 | 0.0648 | 0.0023 | 0.0160 | 0.9134 |
| fog | 0.7351 | 0.7526 | 0.6333 | 0.9441 | 0.0496 | 0.0062 | 0.0196 | 0.9274 |
| synthetic_snow | 0.7012 | 0.6648 | 0.5888 | 0.9184 | 0.0651 | 0.0165 | 0.0151 | 0.9121 |

The largest unsafe-to-safe increase came from synthetic snow, closely followed by Gaussian blur. Confidence decreased for the largest degradations but did not track safety performance monotonically.

## Failure Analysis Summary

The worst unsafe-to-safe cases were creek scenes where ADE20K predicted `earth` on terrain that the conservative RUGD-derived policy treats as high risk. In the failure analysis, `creek_01221` and `creek_01086` showed high mean maximum softmax on unsafe-to-safe pixels, indicating that the model can be confidently wrong under this risk policy.

Representative caption:

```text
confident earth-to-safe failure under a conservative small-AGV policy
```

## Limitations

- Zero-shot ADE20K models have a domain gap to RUGD off-road imagery.
- ADE20K labels are mapped to risk with heuristic keywords.
- The RUGD-to-risk policy is conservative and vehicle-specific, not an official dataset label.
- Maximum softmax probability is only a confidence proxy, not calibrated uncertainty.
- Synthetic weather is image-space corruption, not a physical winter-weather or sensor model.
- No depth estimation, model fusion, terrain dynamics, closed-loop planning, or robot control is included.

## Future Work

- Add calibrated uncertainty or selective prediction.
- Incorporate monocular depth or geometry cues for terrain structure.
- Evaluate physically motivated winter corruptions and real winter off-road data.
- Add vehicle-specific constraints such as wheel size, clearance, slope, and traction.
- Test path-planning behavior using the risk maps rather than only pixel-level metrics.
- Compare with models trained or adapted to off-road semantic segmentation while preserving the no-training baseline as a reference.
