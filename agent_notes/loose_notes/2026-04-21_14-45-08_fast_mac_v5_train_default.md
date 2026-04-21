# fast-mac v5 train renderer default

## Context

The user asked to use the `third_party/fast-mac-gsplat` submodule's v5 API for
training because it should provide faster native batch rendering than Taichi.

The submodule is present at:

```text
third_party/fast-mac-gsplat
```

v5 is built locally:

```text
third_party/fast-mac-gsplat/variants/v5/torch_gsplat_bridge_v5/_C.cpython-311-darwin.so
```

Existing notes already showed v5 beating v6 for current workloads, so this
chunk integrated v5, not v6.

## Changes

- Added `src/train/renderers/fast_mac.py`.
- Added renderer mode `fast_mac`.
- Reused the shared Torch 3D projection path:
  - `project_gaussians_2d`
  - `project_gaussians_2d_batch`
- Converted projected inverse covariance to v5 conics:

```text
conic = [inv_cov_xx, inv_cov_xy, inv_cov_yy]
```

v5 evaluates:

```text
exp(-0.5 * (a dx^2 + 2 b dx dy + c dy^2))
```

- Passed rank-depths to v5 because the shared projection path already depth
  sorts front-to-back. v5 sorts internally with stable Torch argsort, so rank
  depths preserve the existing order.
- Added a new active config:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_8192splats.jsonc
```

- Updated `train_full_dynamic_with_camera_prebake_all_frames.sh` to default to
  that config.
- Left the Taichi config available for comparison.

## Validation

Compile:

```text
uv run python -m py_compile \
  src/train/renderers/fast_mac.py \
  src/train/rendering.py \
  src/train/runtime_types.py \
  src/train/dynamicTokenGS.py
```

Config parse:

```text
renderer=fast_mac
batch_strategy=flatten
effective_gaussians=8192
```

Synthetic adapter smoke:

```text
image (2, 3, 32, 32) finite True mean 0.9813
xyz_grad finite True mean_abs 0.0002466
rgb_grad finite True mean_abs 0.0001746
```

Trainer smoke:

```text
WANDB_MODE=disabled
max_frames=4
steps=2
frames_per_step=2
renderer=fast_mac
loss step 1: 0.3758
loss step 2: 0.3547
```

Diff hygiene:

```text
git diff --check
```

passed.

## Target-shape benchmark

Command:

```bash
uv run python src/benchmarks/mac_renderer_stack_compare.py \
  --height 128 \
  --width 128 \
  --gaussians 8192 \
  --batch-size 8 \
  --warmup 1 \
  --iters 3 \
  --backward \
  --no-include-torch \
  --no-check-outputs \
  --renderers taichi,v5
```

Results:

```text
sparse_sigma_1_5:
  taichi_native   fwd+bwd mean 146.992 ms, per frame 18.374 ms
  metal_v5_native fwd+bwd mean  12.157 ms, per frame  1.520 ms
  v5 speedup vs Taichi: 12.09x

medium_sigma_3_8:
  taichi_native   fwd+bwd mean 158.970 ms, per frame 19.871 ms
  metal_v5_native fwd+bwd mean  10.585 ms, per frame  1.323 ms
  v5 speedup vs Taichi: 15.02x
```

## Caveats

- v5's API is projected-2D. Gradients flow through our Torch 3D projection into
  xyz/scales/quats/opacities/rgb, but depth sorting itself remains piecewise
  constant.
- v5 samples projected pixels according to the fast-mac kernel convention. This
  may not exactly match the old dense renderer's integer-grid convention, so
  compare quality by actual train/eval runs, not pixel-exact dense parity.
- The active config file name says `fast_mac`; the older Taichi config remains
  useful as a slower compatibility/reference path.
