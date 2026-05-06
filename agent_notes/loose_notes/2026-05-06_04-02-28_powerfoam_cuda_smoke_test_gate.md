# PowerFoam CUDA Smoke Test Gate

Date: 2026-05-06 04:02:28 Asia/Ho_Chi_Minh

## Goal

Make the CUDA/Modal PowerFoam lane cheap to validate before spending L40S time:
clone pinned official PowerFoam, apply the minimal dynamic feature-foam fork,
run static and dynamic lanes on the same tiny clip/settings, and keep a local
test that proves the deployment contract is still wired.

## What Changed

- Added `tests/test_powerfoam_cuda_smoke.py`.
- Updated `AGENTS.md` with the CUDA smoke gate and Modal rerun command.
- Updated `research_experiments/dynamic_foam/POWERFOAM_CUDA_MODAL_SMOKE.md`
  with the no-GPU pytest command.
- Added a matched random-init Metal micro config:
  `src/train_configs/local_mac_powerfoam_metal_cuda_micro_match_randominit_64_4f_256cells_5step.jsonc`.
- Added `research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py`
  to write a smoke-scale CUDA-vs-Metal comparison JSON.

The new test has two contracts:

- plan path: `powerfoam_cuda_smoke_runner.py` writes a planned summary that the
  CUDA-smoke verifier accepts without requiring a GPU
- fork scope: `cuda_forks/dynamic_feature_foam.patch` stays limited to upstream
  config and `powerfoam/scene.py`, with no rasterizer/raytrace/CUDA kernel diff

## Verification

Commands run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  --run-id local_plan_check_cuda_micro \
  --output-dir /tmp/powerfoam_cuda_plan_check \
  --frames 4 --size 64 --iterations 5 --points 256 \
  --num-texel-sites 4 --sv-dof 2 --max-gpu-minutes 8 \
  --skip-official-fixture
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  /tmp/powerfoam_cuda_plan_check/summary.json --allow-planned
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  tests/test_powerfoam_cuda_smoke.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_powerfoam_cuda_smoke.py -q
```

Result: `2 passed in 0.56s`.

Additional commands:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_cuda_micro_match_randominit_64_4f_256cells_5step.jsonc
```

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py
```

The first draft Metal config used `init_from_video=true`, which made the metric
comparison unfair against official CUDA's random-bounded init. I replaced it
with random Metal init before saving the comparison artifact.

## Current CUDA Evidence

The strict saved L40S smoke still validates against the current dynamic patch:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

It is a 64px, 4-frame, 5-step smoke. It proves deployment, official static CUDA
baseline execution, dynamic fork execution, coefficient movement, same-camera
time causality, and warm-step timing. It does not prove paper-scale quality or
true F32 feature accumulation through Warp kernels.

Matched CUDA-vs-Metal artifact:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/cuda_vs_metal_summary.json
```

Matched contract: same source clip, 64 px, 4 frames, 5 steps, 256 points/cells,
4 texel sites, SV DoF 2, and random init on the Metal lane. Local Metal eval
PSNR/SSIM/L1 is `5.1222 / 0.0105 / 0.5057`, which is `-0.4183` PSNR,
`-0.0113` SSIM, and `+0.0141` L1 versus strict official static CUDA. This is a
backend smoke sanity record, not a heldout or paper-quality claim.

## Superseded Older Note

The older loose note
`agent_notes/loose_notes/2026-05-06_00-45-00_powerfoam_cuda_modal_smoke.md`
ends with a now-stale completion read saying the selected clean row was
`10.8536 / 0.0766`. Later regular-triangulation OPENCV_FISHEYE evidence
superseded that: selected clean DeepView row is now `12.5099 / 0.1169`, still
below the `13.0 / 0.15` acceptance gate. The CUDA smoke conclusion is unchanged.
