# Depth-aware DoF postprocess prototype

## Context

We wanted a cheap differentiable baseline for camera defocus/focal blur that
does not expand the per-splat renderer backward. The target shape was an
image-space pass after splat rasterization, guided by a rendered or estimated
depth map, with a CPU Torch reference and a Metal-backed path on macOS.

## What landed

- Added `src/train/postprocess_dof.py` with `depth_aware_defocus_blur`.
- The operator uses a fixed maximum gather window and continuous per-pixel
  Gaussian weights from a CoC-like radius:

  ```text
  radius_px = aperture_scale * abs(1 / depth - inv_focus_depth)
  ```

- It supports optional alpha weighting and optional inverse-depth edge gating.
- Default `detach_depth=True` means depth guides the postprocess but does not
  receive the dense windowed image-space gradient.
- Added `tests/test_postprocess_dof.py` for gradient sanity in both detached
  and depth-gradient-enabled modes.
- Added `src/benchmarks/depth_aware_dof_demo.py`, which loads a few local
  images, builds a cheap heuristic depth map, runs CPU and PyTorch MPS, and
  writes comparison strips under `outputs/depth_aware_dof_demo/`.

## What was run

```text
uv run python -m py_compile src/train/postprocess_dof.py tests/test_postprocess_dof.py src/benchmarks/depth_aware_dof_demo.py
uv run python src/train/postprocess_dof.py
uv run python src/benchmarks/depth_aware_dof_demo.py --limit 3 --size 192 --iters 8 --warmup 2
```

`pytest` was not available in the current uv environment, so the two test
functions were imported and called manually. They passed.

## Demo results

The demo used PyTorch MPS as the Metal-backed baseline, not a custom native
Metal shader. At 192px, the MPS path was about 2.2x to 2.6x faster than the CPU
path for this unfold-based prototype, with CPU/MPS max absolute image
differences around `4.17e-7`.

The generated quick depth maps are heuristic only: luminance, vertical image
position, and local contrast. They are useful for visualizing the blur operator
and output panels, not for judging monocular depth quality.

## Backward implication

This does not blow up the per-splat backward in the default path because it is
post-raster and detaches depth. Gradients flow to the rendered RGB image and to
the focus/aperture parameters. The extra cost is an image-space windowed pass
that scales with `H * W * (2R + 1)^2`, not splat count.

If `detach_depth=False`, the blur can propagate gradients into the depth image,
but that is still an image-space path. It should be treated as a separate
experiment because it couples the blur loss back into geometry/depth and can
make optimization less stable.

## Open next steps

- Replace the demo depth heuristic with renderer-provided depth or a real
  depth-estimation probe.
- Add a benchmark row to the renderer harness once the exact training call site
  is chosen.
- Write a custom Metal kernel only after the Torch/MPS path is behaviorally
  useful; the current result is enough to validate the API and gradients.
