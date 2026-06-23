# WinterNav-RUGD

WinterNav-RUGD is a research prototype for confidence-aware traversability mapping on the RUGD off-road robot dataset. It converts official RUGD RGB semantic masks into conservative 0/1/2 risk maps for a small autonomous ground vehicle, then evaluates zero-shot ADE20K segmentation models against that risk ground truth.

No deep model training is performed.

## Hardware

Validated local hardware:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
VRAM: 4 GB
CUDA: available through the project virtual environment
```

## Quick Start

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests
.\.venv\Scripts\python.exe scripts\run_single_image.py --mode ground_truth --sequence creek --filename creek_00001.png --output_dir outputs/phase1_example
.\.venv\Scripts\python.exe scripts\run_single_image.py --mode segformer --sequence creek --filename creek_00001.png --output_dir outputs/phase2_segformer
```

The RUGD dataset should be extracted under:

```text
data/raw/rugd/
```

with these verified paths:

```text
data/raw/rugd/RUGD_frames-with-annotations/RUGD_frames-with-annotations
data/raw/rugd/RUGD_annotations/RUGD_annotations
data/raw/rugd/RUGD_annotations/RUGD_annotations/RUGD_annotation-colormap.txt
```

## Reproduction Commands

Phase 1 ground-truth risk generation:

```powershell
.\.venv\Scripts\python.exe scripts\run_single_image.py --mode ground_truth --sequence creek --filename creek_00001.png --output_dir outputs/phase1_example
```

Phase 3 SegFormer-B0 evaluation:

```powershell
.\.venv\Scripts\python.exe scripts\run_subset_experiment.py --mode segformer_eval --model_name segformer_b0 --subset_size 30 --seed 42 --output_dir outputs/phase4_segformer_b0_cuda
```

Phase 4 SegFormer-B2 comparison:

```powershell
.\.venv\Scripts\python.exe scripts\run_subset_experiment.py --mode segformer_eval --model_name segformer_b2 --subset_size 30 --seed 42 --output_dir outputs/phase4_segformer_b2
```

Phase 5 SegFormer-B2 weather robustness:

```powershell
.\.venv\Scripts\python.exe scripts\run_subset_experiment.py --mode weather_eval --model_name segformer_b2 --subset_size 30 --seed 42 --output_dir outputs/phase5_weather_b2
```

The helper script below prints the validated command set by default and can run it with `-Run`:

```powershell
.\scripts\run_all_validated_experiments.ps1
```

## Key Results

Normal-condition results on the fixed 30-image RUGD subset:

| Model | Device | Accuracy | Balanced Acc. | Macro F1 | High-Risk Recall | Unsafe-to-Safe | Runtime/Image |
|---|---:|---:|---:|---:|---:|---:|---:|
| SegFormer-B0 | CUDA | 0.7174 | 0.6983 | 0.6130 | 0.9196 | 0.0665 | 1.0238 s |
| SegFormer-B2 | CUDA | 0.7455 | 0.7620 | 0.6467 | 0.9446 | 0.0487 | 0.9210 s |

SegFormer-B2 weather robustness:

| Condition | Accuracy | Balanced Acc. | Macro F1 | High-Risk Recall | Unsafe-to-Safe | Mean Confidence |
|---|---:|---:|---:|---:|---:|---:|
| normal | 0.7455 | 0.7620 | 0.6467 | 0.9446 | 0.0487 | 0.9195 |
| low_light | 0.7402 | 0.7566 | 0.6399 | 0.9437 | 0.0485 | 0.9213 |
| gaussian_blur | 0.7198 | 0.7430 | 0.6176 | 0.9329 | 0.0648 | 0.9134 |
| fog | 0.7351 | 0.7526 | 0.6333 | 0.9441 | 0.0496 | 0.9274 |
| synthetic_snow | 0.7012 | 0.6648 | 0.5888 | 0.9184 | 0.0651 | 0.9121 |

## Final Figures

```text
figures/final_model_comparison.png
figures/final_weather_robustness.png
figures/final_failure_case.png
figures/final_pipeline.png
```

## Limitations

- The segmentation models are zero-shot ADE20K models, so there is a domain gap to RUGD off-road terrain.
- The semantic-to-risk conversion is a heuristic policy for a conservative small AGV, not an official RUGD traversability label.
- Synthetic weather is deterministic image corruption, not physical winter weather or sensor simulation.
- Maximum softmax probability is a confidence proxy, not calibrated uncertainty.
- There is no robot control loop, terrain dynamics model, or vehicle-specific traction model yet.

## Project Layout

```text
config/      Label, risk, model, and experiment configuration.
src/         Dataset, traversability, model, weather, evaluation, and visualization code.
scripts/     Single-image, subset, and reproducibility commands.
tests/       Unit tests for core conversion, weather, model helpers, and metrics.
outputs/     Generated experiment outputs.
figures/     Final portfolio figures.
results/     Final research summary.
```
