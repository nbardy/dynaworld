# Taichi OOM and diagnostic guard

## Context

The user reported the previous Taichi 8192-splat run crashed around step 117:

```text
kIOGPUCommandBufferCallbackErrorOutOfMemory
RuntimeError: Invalid buffer size: 32.00 GiB
```

The run was still using:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_taichi_8192splats.jsonc
```

That run started before the train wrapper default was switched to the fast-mac
v5 config. The wrapper now defaults to:

```text
src/train_configs/local_mac_overfit_prebaked_camera_128_4fps_wide_depth_fast_mac_8192splats.jsonc
```

## What failed

There were two issues:

1. Taichi/Metal hit GPU memory pressure during the real training run.
2. The trainer then entered emergency diagnostics and called
   `dense_render_diagnostics(...)`.

For the active shape:

```text
B = 8
G = 8192
H = W = 128
```

the dense diagnostic power tensor would have:

```text
B * G * H * W = 8 * 8192 * 128 * 128 = 1,073,741,824 elements
```

That is about 4 GiB for one float32 tensor, and more once masks, exponentials,
and intermediate tensors are included. The crash reported a 32 GiB invalid
buffer allocation, consistent with the diagnostic path amplifying an already bad
MPS memory situation.

This diagnostic path should never allocate a dense `[B,G,H,W]` tensor for large
8192-splat batches.

## Fix

Added safety guards in `src/train/debug_metrics.py`:

- `DEFAULT_PIXEL_DIAG_MAX_WORK_ITEMS = 64_000_000`
- skip pixel-power / alpha-pre diagnostics above that threshold
- log:

```text
RenderDiag/PixelWorkItems
RenderDiag/PixelDiagSkipped
```

- `DEFAULT_TILE_DIAG_MAX_GAUSSIANS = 4096`
- skip tile-overlap diagnostics above that threshold
- log:

```text
TileDiag/Skipped
```

Geometry diagnostics still run:

```text
xyz/scale ranges
camera z range
front/behind splat counts
projected means range
determinant/inv-cov ranges
render nonfinite count when renders are available
```

## Validation

Compile passed:

```text
uv run python -m py_compile src/train/debug_metrics.py src/train/dynamicTokenGS.py
```

Synthetic large-shape diagnostic smoke:

```text
B=8, G=8192, H=W=128
PixelDiagSkipped 1.0
TileDiagSkipped 1.0
has PowerMax False
RenderNonfiniteCount 0.0
```

`git diff --check` passed.
