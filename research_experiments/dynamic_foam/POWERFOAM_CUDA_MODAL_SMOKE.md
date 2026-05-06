# PowerFoam CUDA Modal Smoke

Date: 2026-05-06

Purpose: run the pinned official CUDA/Warp PowerFoam code and a minimal
dynamic feature-foam fork on the same tiny clip/settings, then save comparable
JSON results for Metal-vs-CUDA follow-up. This is a fast deployment and
regression smoke, not a paper-quality benchmark.

## What It Runs

- Official repo: `https://github.com/theialab/powerfoam`
- Pinned commit: `96392252ebd0059fe6ca98881b62e12295d9242f`
- GPU target: Modal `L40S`
- Clip: `test_data/test_video_small_128_4fps.mp4`
- Default preset: `micro_clip_64_4f_5step`
- Static lane: upstream `PowerfoamScene` with official CUDA/Warp renderer
- Dynamic lane: copied upstream checkout plus
  `cuda_forks/dynamic_feature_foam.patch`

The dynamic fork deliberately keeps upstream geometry, adjacency, SV color
query, and rasterizer/raytracer intact. The first small tweak is a
time-conditioned residual on `texel_sv_rgb`, keyed by `camera.time_index`.
True F32 feature accumulation through Warp is a later kernel-level fork, not
this cheap smoke.

## Commands

For a durable rerun, prefer a datetime run id such as
`2026-05-06_l40s_micro64_4f5`. The strict micro preset is the canonical
cheap CUDA time-causality proof. The older `latest` run is only a 128px
deployment reference.

Plan only, no GPU spend:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_time_causality_rerun \
  --skip-official-fixture \
  --fixed-black-background
```

Execute on one L40S:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_time_causality_rerun \
  --max-gpu-minutes 8 \
  --skip-official-fixture \
  --fixed-black-background
```

Execute the same micro smoke with the scene-side dynamic-geometry fork as a
third lane:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_dynamic_geometry_micro_rerun \
  --max-gpu-minutes 8 \
  --skip-official-fixture \
  --fixed-black-background \
  --dynamic-geometry
```

Execute the same strict micro smoke with the official CUDA/Warp parity fixture
included when the extra setup time is acceptable:

```bash
uv run --with modal modal run research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute \
  --preset micro_clip_64_4f_5step \
  --run-id cuda_micro_blackbg_fixture_rerun \
  --max-gpu-minutes 10 \
  --fixed-black-background
```

Validate returned JSON:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/{run_id}/summary.json
```

Require the fixture for fixture-included reruns:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/{run_id}/summary.json \
  --require-official-fixture
```

Fast no-GPU regression gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_powerfoam_cuda_smoke.py -q
```

This exercises the plan path, validator shape, upstream pin, current dynamic
patch SHA, strict micro settings, the CUDA-vs-Metal comparison writer contract,
the Modal deploy wrapper flags, the optional fixture-required validator mode,
and the guard that the CUDA fork remains an appearance-side scene/config patch
rather than a Warp kernel patch.

Matched local Metal smoke, same source clip/64px/4f/5step/256 cells/4 texel
sites/SV DoF 2, random init, fixed black background:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/powerfoam-metal \
  WANDB_MODE=disabled .venv/bin/python src/train/train_powerfoam_metal.py \
  src/train_configs/local_mac_powerfoam_metal_cuda_micro_match_randominit_64_4f_256cells_5step.jsonc
```

Write the smoke-scale CUDA-vs-Metal comparison JSON:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/compare_powerfoam_cuda_metal_smoke.py
```

If the official fixture was copied back, run the existing local parity nodes:

```bash
PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest \
  tests/test_powerfoam_direct.py::test_powerfoam_direct_matches_official_cuda_fixture_if_present \
  tests/test_powerfoam_direct.py::test_powerfoam_metal_matches_official_cuda_fixture_shared_backward_if_present \
  -q -rs
```

## Artifact Contract

Main output:

```text
outputs/powerfoam_cuda_smokes/{run_id}/summary.json
```

The summary records:

- official repo URL and commit
- dynamic patch SHA256
- CUDA/Torch/Warp/GPU host info
- clip SHA256, frame count, render size
- same settings for both lanes
- per-lane eval metrics and phase timings
- dynamic-minus-static PSNR/SSIM and speed ratio

The Modal wrapper returns small JSON files only. It does not copy model
checkpoints by default because this lane is meant to minimize GPU time and
local artifact churn.

## Latest Results

Exact fixed-black dynamic-geometry micro smoke:

```text
outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json
```

Validated on 2026-05-06 with:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json \
  --require-dynamic-geometry
```

Settings: `L40S`, 4 frames, 64 px, 5 steps, 256 points, 4 texel sites,
SV DoF 2, `--skip-official-fixture`, `--fixed-black-background`,
`--dynamic-geometry`.

| Lane | Status | Eval PSNR | Eval SSIM | Eval L1 | Warm Step Mean |
|---|---|---:|---:|---:|---:|
| `official_static_cuda` | ok | 5.5640 | 0.0284 | 0.4901 | 8.53 ms |
| `dynamic_feature_foam_cuda` | ok | 5.5833 | 0.0288 | 0.4887 | 9.17 ms |
| `dynamic_geometry_foam_cuda` | ok | 5.5910 | 0.0291 | 0.4882 | 11.64 ms |

The feature fork still only changes RGB: `time_alpha_delta_mean=0.0` and
`same_camera_support_delta_mean=0.0`. The geometry fork records actual
scene/support motion: `dynamic_center_delta_mean=0.0005558`,
`dynamic_radius_delta_mean=0.00000796`,
`dynamic_height_delta_mean=0.0007690`,
`time_alpha_delta_mean=0.002022`, and
`same_camera_support_delta_mean=0.003174`. This is a smoke-scale deployment
proof, not a real-quality dynamic-geometry benchmark.

Exact fixed-black micro smoke:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json
```

Validated on 2026-05-06 with the stricter verifier. Settings: `L40S`, 4
frames, 64 px, 5 steps, 256 points, 4 texel sites, SV DoF 2,
`--skip-official-fixture`, `--fixed-black-background`.

| Lane | Status | Eval PSNR | Eval SSIM | Eval L1 | Cold Step Mean | Warm Step Mean |
|---|---|---:|---:|---:|---:|---:|
| `official_static_cuda` | ok | 5.5640 | 0.0284 | 0.4901 | 1510.45 ms | 8.31 ms |
| `dynamic_feature_foam_cuda` | ok | 5.5833 | 0.0288 | 0.4887 | 160.52 ms | 9.09 ms |

Dynamic-minus-static: `+0.0193` PSNR, `+0.000454` SSIM. The dynamic fork
records a same-camera time-causality probe:
`dynamic_time_rgb_delta_mean=0.00006899`,
`dynamic_time_rgb_delta_max=0.0009796`, and camera times cover `[0.0, 1.0]`.

Earlier random-background strict micro smoke:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_time_causality_20260506/summary.json
```

Validated on 2026-05-06 with the stricter verifier. Settings: `L40S`, 4
frames, 64 px, 5 steps, 256 points, 4 texel sites, SV DoF 2,
`--skip-official-fixture`. This saved run used random training backgrounds, so
it is not an exact background-policy match for the local black-background Metal
micro config. Use `cuda_micro_blackbg_20260506` for exact CUDA-vs-Metal setting
parity.

| Lane | Status | Eval PSNR | Eval SSIM | Eval L1 | Cold Step Mean | Warm Step Mean |
|---|---|---:|---:|---:|---:|---:|
| `official_static_cuda` | ok | 5.5405 | 0.0218 | 0.4916 | 1204.90 ms | 6.93 ms |
| `dynamic_feature_foam_cuda` | ok | 5.5487 | 0.0221 | 0.4911 | 132.05 ms | 7.17 ms |

Dynamic-minus-static: `+0.00828` PSNR, `+0.000345` SSIM. The dynamic fork
now also records a same-camera time-causality probe:
`dynamic_time_rgb_delta_mean=0.00019385`,
`dynamic_time_rgb_delta_max=0.0026973`, and camera times cover `[0.0, 1.0]`.
That proves the time-conditioned branch changes rendered RGB on this smoke; it
does not make the fork a temporal-quality benchmark.

The older 128px reference smoke remains useful for rough deployment metrics:

Run:

```text
outputs/powerfoam_cuda_smokes/latest/summary.json
```

Settings: `L40S`, 8 frames, 128 px, 20 steps, 512 points,
4 texel sites, SV DoF 4. Both lanes used the same exported clip SHA256
`f10c67ee46f4675d6b9b89ea625302b31b2e3043244260092873698b5e5bd6da`.

| Lane | Status | Eval PSNR | Eval SSIM | Eval L1 | Cold Step Mean | Warm Step Mean |
|---|---|---:|---:|---:|---:|---:|
| `official_static_cuda` | ok | 6.0284 | 0.0577 | 0.4551 | 314.84 ms | not in `summary.json`; `modal_return` lane metrics derive 8.23 ms |
| `dynamic_feature_foam_cuda` | ok | 6.0773 | 0.0610 | 0.4521 | 43.13 ms | not in `summary.json`; `modal_return` lane metrics derive 8.79 ms |

Dynamic-minus-static: `+0.0488` PSNR, `+0.00335` SSIM. The dynamic
time-conditioned parameter moved (`abs_mean=0.0290`, `abs_max=0.0573`) and
camera times covered `[0.0, 1.0]` across 8 frames.

Schema caveat: `latest` predates the stricter warm-timing/time-causality
summary fields and intentionally fails the new
`dynamic_time_changes_rendered_rgb` and `warm_timing_recorded` checks. Use
`cuda_micro_time_causality_20260506` or a fresh dated run when citing the
strict CUDA comparison.

Timing caveat: the cold step mean includes first-step Warp/CUDA JIT and
run-order cache effects. Warm step mean excludes step 0 and is the saner
smoke-level timing read.

## CUDA-vs-Metal Smoke Comparison

Current matched random-init fixed-black Metal comparison:

```text
outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/cuda_vs_metal_summary.json
```

The matched contract is true for source clip, frame count, render size, step
count, point/cell count, texel-site count, SV DoF, random init, and fixed black
background. The local Metal lane reports eval PSNR/SSIM/L1
`5.1222 / 0.0105 / 0.5057`. Against the fixed-black official static CUDA lane,
this is `-0.4418` PSNR, `-0.0179` SSIM, and `+0.0155` L1. Treat this only as
a smoke-scale cross-backend sanity record; CUDA lanes run official upstream
PowerFoam/Warp on Modal L40S, while the Metal lane runs the local Metal trainer
on MPS.

The official CUDA/Warp parity fixture was generated and copied to:

```text
research_experiments/dynamic_foam/fixtures/powerfoam_tiny_height_sv_official_camera_official_v1.json
```

Follow-up parity status: the local Direct/Metal official parity pytest nodes
now pass against this copied CUDA fixture. The important fix was matching
upstream's effective raster texture temperature (`10.0`) and comparing the
stable shared-backward channels rather than tiny geometry channels pruned by
the pinned CUDA/Warp backward path.
