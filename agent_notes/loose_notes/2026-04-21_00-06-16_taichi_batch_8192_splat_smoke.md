# Taichi Batch 8192-Splat Smoke

## Context

The Taichi fork now exposes native `rasterize_batch` for `[B, G, 7]` packed
2D Gaussians, `[B, G, 1]` depth keys, and `[B, G, C]` features. The parent
Dynaworld train adapter had already been updated locally to call that native
batch path from `render_gaussian_frames(..., mode="taichi")`.

## Confirmed Train Adapter State

- `src/train/renderers/taichi.py` imports `project_gaussians_2d_batch`.
- `project_for_taichi_axis_batch(...)` packs `[B, G, 7]`.
- `render_taichi_3dgs_batch(...)` calls `taichi_splatting.rasterizer.rasterize_batch`.
- `src/train/rendering.py` routes `mode == "taichi"` batch rendering to
  `render_taichi_3dgs_batch(...)`.
- Dense/tiled non-Taichi fallback still loops or uses dense batch as before.

## New Config

Added:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi_8192splats.jsonc
```

This is the 128px/4fps wide-depth Taichi run with:

- `tokens = 128`
- `gaussians_per_token = 64`
- total explicit Gaussians = `8192`
- `frames_per_step = 4`
- `renderer = "taichi"`

Dense renderer diagnostics are disabled in this config because they still run
dense/common diagnostic math and would be very expensive at 8192 splats.

## Validation

Compile:

```bash
uv run python -m py_compile \
  src/train/renderers/taichi.py \
  src/train/rendering.py \
  src/train/dynamicTokenGS.py
```

passed.

Synthetic batched adapter smoke:

- device: `mps`
- output shape: `(2, 3, 32, 32)`
- output finite: true
- xyz gradient finite: true
- RGB gradient finite: true

One-step 8192-splat train smoke:

- loaded 46 frames from the 128px/4fps bake
- ran `128 x 64 = 8192` explicit Gaussians
- renderer: `taichi`
- four frames per step
- finite loss: `0.1578`

Three-step 8192-splat train smoke:

- losses: `0.1791`, `0.1587`, `0.1817`
- no NaNs or non-finite render/loss
- wall time including Python/Taichi startup and first Metal compile: `34.95s`
- tqdm showed first step dominated by compile, later steps were much faster

## Run Command

```bash
./src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh \
  src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi_8192splats.jsonc
```

## Caveats

- The first step in a fresh Python process includes Taichi/Metal compile.
- This remains the simple `metal_reference` / `pixel_reference` renderer, not
  the final fused kernel.
- More splats per token gives more primitive capacity, but those splats still
  share the same token latent. If detail remains weak, compare against more
  tokens with fewer splats/token, not only `gaussians_per_token` scaling.
