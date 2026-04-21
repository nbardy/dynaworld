# Taichi Renderer Training Adapter

## Context

The current 128px/4fps known-camera baseline is stable after the renderer finite-depth guard and the wider Gaussian head depth range, but the dense PyTorch renderer is slow at 128px. The local fork under `third_party/taichi-splatting/` already exposes a Torch autograd rasterizer, and the benchmark code already had a 3D-to-Taichi 2D packed Gaussian adapter.

The goal for this chunk was to integrate Taichi as an explicit trainer renderer mode without changing model capacity yet. The 64-splats-per-token experiment should be run after this renderer baseline is validated, otherwise renderer differences and capacity differences will be mixed.

## Implementation

- Added `renderer: "taichi"` as an explicit mode.
- Added `src/train/renderers/taichi.py`.
- Reused the benchmark projection shape:
  - call our Torch `project_gaussians_2d`
  - eigendecompose the 2D covariance
  - pack `[x, y, axis_x, axis_y, sigma_major, sigma_minor, opacity]`
  - use rank-depth because `project_gaussians_2d` already sorts front-to-back
  - call `taichi_splatting.rasterizer.rasterize`
  - blend white background with `raster.image_weight`
- Kept Taichi frame batching simple: `render_gaussian_frames` loops frames for non-dense renderers.
- Added config `src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi.jsonc`.

## Validation

- `uv run python -m py_compile src/train/runtime_types.py src/train/renderers/taichi.py src/train/rendering.py src/train/dynamicTokenGS.py`
- Parsed the new config successfully.
- Synthetic MPS Taichi render and backward were finite:
  - output shape `(3, 32, 32)`
  - render finite true
  - xyz/rgb grads finite true
- One-step trainer smoke with Taichi and one frame completed.
- One-step trainer smoke with Taichi and the real four-frame batch path completed:
  - 46 frames loaded from the 128px/4fps camera JSON
  - 512 splats
  - renderer `taichi`
  - loss finite
- Dense/Taichi synthetic render comparison was close enough for a first adapter:
  - dense mean `0.99113`
  - taichi mean `0.99124`
  - MAE `0.00381`
  - max difference `0.19434`

## Caveats

- Taichi is not batched yet in the trainer; it loops per frame.
- The Taichi backward is `pixel_reference`, so it is correctness-oriented, not necessarily the final fast training kernel.
- The adapter uses float32 on Metal.
- The rank-depth trick preserves front-to-back ordering from our existing projection; if we later need exact real camera z in Taichi diagnostics, return sorted z from the projection boundary instead.
- Renderer diagnostics still use the dense/common diagnostic path. That is okay for geometric health metrics, but Taichi-specific tile/visibility metrics are not wired into trainer logging yet.

## Depth-Range Fix Reminder

The earlier "depth range / camera normalization" fix was not true normalization. It made the Gaussian head depth range configurable and gave the 128px config `z_max=5.0` instead of the old hard cap of `2.5`.

That matters because the 128px DUSt3R camera path translates late cameras far enough that the old initial Gaussian slab could fall behind or too close to the camera for late-frame windows. Metrics showed the first-order symptom:

- `FrontGaussiansMin=0` on late windows
- `NearOrBehindGaussiansMean` near the full 512 splats
- nearly all slots with no alpha support/contribution

The wide-depth config gives the model representational room to put splats in front of those late cameras. A proper long-term fix would normalize each scene/camera path into a canonical training volume so this is not hand-tuned per clip.
