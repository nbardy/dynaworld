# PowerFoam CUDA Smoke Reverification

Date: 2026-05-06 04:54:00 Asia/Ho_Chi_Minh

## Context

The goal for this work chunk was to answer whether we have the fast CUDA
deployment/test case requested for PowerFoam:

- clone the pinned official/base PowerFoam CUDA/Warp repo
- run the official static CUDA baseline on the same tiny clip/settings
- copy/fork that checkout with the smallest dynamic time-conditioned feature
  foam tweak
- save comparable CUDA results and compare them against the matched Metal
  micro smoke
- keep Modal L40S spend minimal

## Current State

The lane already exists and is passing:

- `research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py` performs
  the clone/apply/run/log flow.
- `research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py` wraps it in
  a Modal `L40S` runner with plan-only mode by default.
- `research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch`
  adds the small dynamic `texel_sv_rgb` residual to upstream `powerfoam/scene.py`
  without touching Warp raster/raytrace kernels.
- `tests/test_powerfoam_cuda_smoke.py` protects the no-GPU plan contract and
  the small-patch boundary.
- `research_experiments/dynamic_foam/POWERFOAM_CUDA_MODAL_SMOKE.md`,
  `BASELINES.md`, and
  `research_experiments/dynamic_foam/external_powerfoam_artifact_handoff.json`
  point to the saved artifacts and rerun commands.

The strict dated CUDA artifact to cite is:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

It is the canonical minimal L40S proof because it includes warm-step timing and
rendered same-camera time-causality fields. The older `latest` 128px smoke is
only a deployment reference and predates those strict validator fields.

## Verification This Turn

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_powerfoam_cuda_smoke.py -q
```

Result: `2 passed in 0.42s`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

Result: `ok=true`. The verifier confirmed the pinned official commit, current
dynamic patch SHA, CUDA host metadata (`NVIDIA L40S`, Torch `2.11.0+cu130`,
Warp `1.10.0`), both CUDA lanes, warm timing, coefficient movement, and
rendered RGB time-causality.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py
```

Result: rewrote
`outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/cuda_vs_metal_summary.json`
with `status=ok`.

## Numbers To Reuse Carefully

Strict micro CUDA, same clip/settings: 4 frames, 64 px, 5 steps, 256 points,
4 texel sites, SV DoF 2.

Official static CUDA:

- eval PSNR/SSIM/L1: `5.5405 / 0.0218 / 0.4916`
- warm step mean: `6.93 ms`

Dynamic feature-foam CUDA:

- eval PSNR/SSIM/L1: `5.5487 / 0.0221 / 0.4911`
- warm step mean: `7.17 ms`
- rendered time-causality: `dynamic_time_rgb_delta_mean=0.00019385`,
  `dynamic_time_rgb_delta_max=0.0026973`

Matched random-init Metal micro:

- eval PSNR/SSIM/L1: `5.1222 / 0.0105 / 0.5057`

This is a smoke-scale backend sanity record, not paper-scale quality. The
dynamic CUDA fork is appearance-side time-conditioned SV RGB, not a true F32
feature-accumulating CUDA/Warp kernel fork yet.

## Open Follow-Up

The cheap CUDA baseline/fork path is done. The next CUDA-side representation
step, if needed, is to fork upstream kernels for actual F32 feature
accumulation rather than only time-conditioning `texel_sv_rgb`.

