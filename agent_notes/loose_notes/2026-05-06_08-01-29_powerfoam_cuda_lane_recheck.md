# PowerFoam CUDA Lane Recheck

Date: 2026-05-06 08:01:29 Asia/Ho_Chi_Minh

## Goal

Recheck the requested fast CUDA deployment lane for PowerFoam:

- clone the pinned official/base PowerFoam CUDA/Warp repo,
- run official static CUDA on the same tiny clip/settings,
- copy/fork that checkout for a small dynamic time-conditioned feature-foam
  representation tweak,
- save comparable baseline/dynamic/Metal JSON results,
- keep Modal L40S compute minimal.

## Current State

This lane already exists and is wired:

- `research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py` is the
  Modal L40S entrypoint.
- `research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py` handles
  dry-run planning, official clone, dynamic fork copy, patch application, lane
  execution, and summary writing.
- `research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch`
  is a real upstream patch, not a placeholder. It adds a Gaussian
  time-basis residual on upstream `texel_sv_rgb`, keyed by `camera.time_index`,
  without touching Warp raster/raytrace kernels.
- `tests/test_powerfoam_cuda_smoke.py` guards the no-GPU plan contract, the
  patch boundary, and CUDA-vs-Metal comparison JSON contract.
- `research_experiments/dynamic_foam/POWERFOAM_CUDA_MODAL_SMOKE.md`,
  `BASELINES.md`, `TODO/powerfoam_full_reproduction_todo.md`, and
  `research_experiments/dynamic_foam/external_powerfoam_artifact_handoff.json`
  route to the saved runs and rerun commands.

## Verification This Turn

No new Modal GPU work was launched. The strict saved L40S artifact was already
present and still validates against the current patch SHA.

Commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_cuda_smoke.py -q
```

Result: `3 passed in 0.45s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

Result: `ok=true`. It confirmed pinned official commit
`96392252ebd0059fe6ca98881b62e12295d9242f`, current dynamic patch SHA,
CUDA host metadata (`NVIDIA L40S`, Torch `2.11.0+cu130`, Warp `1.10.0`),
official static CUDA, dynamic feature-foam CUDA, warm timing, coefficient
movement, and rendered same-camera time-causality.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  --run-id local_plan_check_latest \
  --output-dir /tmp/powerfoam_cuda_plan_check_latest \
  --frames 4 --size 64 --iterations 5 --points 256 \
  --num-texel-sites 4 --sv-dof 2 --max-gpu-minutes 8 \
  --skip-official-fixture
```

Then:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  /tmp/powerfoam_cuda_plan_check_latest/summary.json --allow-planned
```

Result: `ok=true`; the no-GPU plan contains the official clone/fetch/checkout
steps and the dynamic patch apply step.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py
```

Result: rewrote
`outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/cuda_vs_metal_summary.json`
with `status=ok`.

## Reuse Guidance

For the cheapest current CUDA proof, cite:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

Settings: 4 frames, 64 px, 5 steps, 256 points, 4 texel sites, SV DoF 2,
Modal L40S, official fixture skipped.

The dynamic fork is intentionally appearance-side time-conditioned SV RGB. It
does prove deployability, same-clip comparison, trainable dynamic coefficients,
and rendered RGB dependence on `camera.time_index`. It does not yet implement
true F32 feature accumulation/backward inside the upstream CUDA/Warp kernels.

## Fixed-Black Follow-Up

The first strict micro run used CUDA's random training background, while the
matched local Metal micro config uses fixed black background. I updated the
comparison contract to check background policy explicitly, so the old
`cuda_micro_time_causality_20260506/cuda_vs_metal_summary.json` now reports
`status=mismatch` as intended.

I then launched the exact fixed-black Modal smoke:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_blackbg_20260506 \
  --max-gpu-minutes 8 \
  --skip-official-fixture \
  --fixed-black-background
```

Artifact:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json
```

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json
```

Result: `ok=true`.

Fixed-black numbers:

- Official static CUDA: eval PSNR/SSIM/L1 `5.5640 / 0.0284 / 0.4901`, warm
  step `8.31 ms`.
- Dynamic feature-foam CUDA: eval PSNR/SSIM/L1 `5.5833 / 0.0288 / 0.4887`,
  warm step `9.09 ms`.
- Dynamic minus static: `+0.0193` PSNR, `+0.000454` SSIM.
- Dynamic time probe: RGB delta mean/max `0.00006899 / 0.0009796`.

Comparison:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py \
  --cuda-summary outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json \
  --output outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/cuda_vs_metal_summary.json
```

Result: `status=ok`, with `same_fixed_black_background=true`. The comparator
default was moved to this fixed-black run.
