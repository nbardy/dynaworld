# PowerFoam CUDA Dynamic Geometry Smoke

## Goal

Close the P0.4 gap without pretending the existing appearance-side CUDA fork was
dynamic geometry. The required proof was a separate upstream PowerFoam CUDA fork
where time changes scene geometry before the official Warp renderer, plus a
strict verifier that rejects RGB-only time causality.

## What Changed

- Added `research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch`.
  It patches pinned upstream PowerFoam at
  `96392252ebd0059fe6ca98881b62e12295d9242f`.
- The patch stays scene/config-side:
  - adds `dynamic_geometry_foam`;
  - adds time-basis coefficients for centers, radii, quaternions, and heights;
  - decodes `points(t)`, `radii(t)`, `quaternions(t)`, and `texel_height(t)`
    before calling the existing CUDA/Warp rasterizer/raytracer;
  - does not patch `powerfoam/rasterize.py`, `powerfoam/raytrace.py`, or CUDA
    kernels.
- Updated `powerfoam_cuda_smoke_runner.py` and
  `modal_powerfoam_cuda_smoke.py` so `--dynamic-geometry` runs three lanes on
  the same exported tiny clip:
  - `official_static_cuda`
  - `dynamic_feature_foam_cuda`
  - `dynamic_geometry_foam_cuda`
- Updated `verify_powerfoam_cuda_smoke_results.py` with
  `--require-dynamic-geometry`. Strict mode now requires:
  - geometry patch hash matches the checked-in patch;
  - geometry lane status is `ok`;
  - geometry coefficients are nonzero;
  - time changes scene points;
  - time changes rendered alpha;
  - time changes alpha/support.

## Verification

Local syntax and patch checks:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/powerfoam_cuda_smoke_runner.py \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  tests/test_powerfoam_cuda_smoke.py

git -C /tmp/powerfoam_upstream_963 apply --check \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/dynamic_foam/cuda_forks/dynamic_feature_foam.patch

git -C /tmp/powerfoam_upstream_963 apply --check \
  /Users/nicholasbardy/git/gsplats_browser/dynaworld/research_experiments/dynamic_foam/cuda_forks/dynamic_geometry_foam.patch
```

Local no-GPU contract:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train uv run --with pytest python \
  -m pytest -p no:cacheprovider tests/test_powerfoam_cuda_smoke.py -q
```

Result: `8 passed`.

Old appearance-only strict rejection:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_micro_blackbg_20260506/summary.json \
  --require-dynamic-geometry
```

Result: failed as intended because the saved appearance-only summary has no
geometry lane and has `dynamic_time_alpha_delta_mean=0.0`.

Modal execution:

```bash
PYTHONDONTWRITEBYTECODE=1 uv run --with modal modal run \
  research_experiments/dynamic_foam/modal_powerfoam_cuda_smoke.py \
  --execute --preset micro_clip_64_4f_5step \
  --run-id cuda_dynamic_geometry_micro_20260506 \
  --max-gpu-minutes 8 --skip-official-fixture \
  --fixed-black-background --dynamic-geometry
```

Strict executed verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/dynamic_foam/verify_powerfoam_cuda_smoke_results.py \
  outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json \
  --require-dynamic-geometry
```

Result: `ok: true`.

## Measured Artifact

Saved summary:

```text
outputs/powerfoam_cuda_smokes/cuda_dynamic_geometry_micro_20260506/summary.json
```

Metrics:

```text
official_static_cuda       PSNR/SSIM/L1 5.5640 / 0.0284 / 0.4901, warm step 8.53 ms
dynamic_feature_foam_cuda  PSNR/SSIM/L1 5.5833 / 0.0288 / 0.4887, warm step 9.17 ms
dynamic_geometry_foam_cuda PSNR/SSIM/L1 5.5910 / 0.0291 / 0.4882, warm step 11.64 ms
```

Feature-only time branch:

```text
time_alpha_delta_mean 0.0
same_camera_support_delta_mean 0.0
time_rgb_delta_mean 0.00006899
```

Geometry time branch:

```text
dynamic_center_delta_mean 0.0005558
dynamic_radius_delta_mean 0.00000796
dynamic_height_delta_mean 0.0007690
dynamic_quaternion_delta_mean 0.005564
time_alpha_delta_mean 0.002022
same_camera_support_delta_mean 0.003174
time_rgb_delta_mean 0.001047
```

## Interpretation

This completes the minimal P0.4 CUDA dynamic-geometry smoke. It proves the
pinned official CUDA/Warp implementation can host a scene-side dynamic geometry
fork without rewriting the renderer, and it gives future agents a strict gate
that separates repaint from motion.

It is not a real dynamic-quality benchmark. The run is only 64 px, 4 frames,
5 steps, and 256 points. The next quality question belongs in P0.3/P1: run a
Metal dynamic-geometry training artifact with motion-vs-repaint controls and
then decide whether to spend CUDA time beyond smokes.
