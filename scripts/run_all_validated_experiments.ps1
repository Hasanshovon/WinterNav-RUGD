param(
    [switch]$Run
)

$ErrorActionPreference = "Stop"

$python = ".\.venv\Scripts\python.exe"
$commands = @(
    @{
        Phase = "Tests"
        Note = "Run the unit-test suite."
        Command = "$python -m unittest discover -s tests"
    },
    @{
        Phase = "Phase 1"
        Note = "Generate one verified RUGD-derived ground-truth risk example."
        Command = "$python scripts\run_single_image.py --mode ground_truth --sequence creek --filename creek_00001.png --output_dir outputs\repro_phase1_example"
    },
    @{
        Phase = "Phase 3/4"
        Note = "SegFormer-B0 CUDA benchmark on the validated 30 selected pairs."
        Command = "$python scripts\run_subset_experiment.py --mode segformer_eval --model_name segformer_b0 --subset_size 30 --seed 42 --output_dir outputs\repro_segformer_b0_cuda"
    },
    @{
        Phase = "Phase 4"
        Note = "SegFormer-B2 CUDA benchmark on the same validated 30 selected pairs."
        Command = "$python scripts\run_subset_experiment.py --mode segformer_eval --model_name segformer_b2 --subset_size 30 --seed 42 --output_dir outputs\repro_segformer_b2_cuda"
    },
    @{
        Phase = "Phase 4 smoke only"
        Note = "UPerNet ConvNeXt-Tiny is documented as smoke-test-only, not part of the main benchmark."
        Command = "$python scripts\run_subset_experiment.py --mode segformer_eval --model_name upernet_convnext_tiny --subset_size 3 --seed 42 --output_dir outputs\repro_upernet_smoke"
    },
    @{
        Phase = "Phase 5"
        Note = "SegFormer-B2 deterministic synthetic-weather robustness evaluation."
        Command = "$python scripts\run_subset_experiment.py --mode weather_eval --model_name segformer_b2 --subset_size 30 --seed 42 --output_dir outputs\repro_weather_b2"
    }
)

Write-Host "WinterNav-RUGD validated reproducibility commands"
Write-Host "Mask2Former is intentionally excluded because strict validation found newly initialized trainable parameters."
Write-Host ""

foreach ($item in $commands) {
    Write-Host "[$($item.Phase)] $($item.Note)"
    Write-Host $item.Command
    if ($Run) {
        Invoke-Expression $item.Command
    }
    Write-Host ""
}

if (-not $Run) {
    Write-Host "Dry run only. Re-run with -Run to execute these validated commands."
}
