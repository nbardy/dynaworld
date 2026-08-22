# Browser StableGS-Inspired Ablation Stack

Date: 2026-08-21

## Context

The browser trajectory-3DGS trainer had real camera-stress diagnostics but no
camera-stress loss. Small novel-camera movements still exposed translucent
clouds. The user asked to implement the paper-backed controls, keep them
independently ablatable, support glass without an external depth model, and
retain the fast fused shader path.

## Work Completed

- Added reset-time pixel-filter specialization: legacy floor or compensated
  2D Mip.
- Added reset-time opacity specialization: coupled or dual geometry/material.
- Reused splat lane 11 for material opacity without widening the 24-float ABI.
- Kept the 12-float projected-gradient record by carrying material gradient in
  `colorPad.w` and mean-alpha density evidence in `screen1.z`.
- Added a fused geometry color/transmittance/depth path to the existing sorted
  raster loop.
- Added fused checkpoint-block adjoints for geometry color and normalized
  expected depth.
- Added periodic prior-free paired-camera reference depth and source-side
  reprojection gradients in the same queue submission.
- Added train-only pair selection from seed-frustum co-visibility and moderate
  camera rotation, with explicit fallback telemetry.
- Added CPU multilayer-ray and second-layer-mass diagnostics.
- Added UI controls, status descriptions, worker transport, memory estimates,
  benchmark CLI options, and regression tests.

## Important Backtracks

1. Dual opacity initially reused `harmonicPad.w`, which had also carried mean
   alpha into densification. The staged update now reads mean alpha from
   projected gradient `screen1.z`; otherwise enabling dual opacity would have
   silently changed topology behavior.
2. The first material-logit trust region was `[-8, 8]`. With the `+log(99)`
   compatibility bias, the lower endpoint was still about 3.2% opacity. It is
   now `[-16, 8]`, permitting approximately `1.1e-5` for real appearance
   transparency.
3. A camera-stress metric is not camera-stress training. The new paired-depth
   switch changes the objective; the multilayer metrics remain diagnostic.
4. Mip-Splatting contains separate 2D and 3D mechanisms. Only the 2D
   determinant compensation is present and named accordingly.

## Current Model

The implementation is StableGS-inspired and prior-free. Geometry opacity is
the existing base opacity. Appearance opacity multiplies it by a learned
material gate. Geometry color and geometry expected depth share the same
front-to-back traversal as the appearance raster. Reference depth is rendered
only at the configured cadence.

The pair contract is weaker than the paper: the browser bundle has no COLMAP
track table or homography inlier fraction, so seed-frustum overlap substitutes
for feature co-visibility. The depth gradient is one-sided per event and
rotates direction over time. This is useful, but it must not be called exact
StableGS.

## Verification

`npm test` in `web/dynaworld_browser_trainer` passes all 193 tests. JavaScript
syntax checks and `git diff --check` pass. The tests cover baseline equivalence,
the new math helpers, train-only pair selection, worker/UI transport, CPU
diagnostics, buffer estimates, and the no-readback train-step contract.

The current environment did not provide a command-line WGSL compiler or Bun
WebGPU. The hidden-browser benchmark dependency was not installed, and the
in-app browser automation surface rejected the localhost URL. Therefore live
Apple shader compilation, finite-difference geometry parity, and throughput
remain explicit runtime gates.

## Resource Estimate

For 8,192 capacity splats with packed projection VJP:

- 96x72: 23.45 MiB baseline, 53.14 MiB complete geometry stack.
- 384x288: 191.13 MiB baseline, 278.67 MiB complete geometry stack.
- the 384 geometry checkpoint is 108 MiB and remains below the portable
  128 MiB storage-binding floor by increasing checkpoint stride to 64.

These are static estimates, not measured GPU residency.

## Next Experiments

Run A0 baseline, A1 2D Mip, A2 dual plus geometry color, A3 dual/color/depth,
and A4 combined under identical initialization, camera/time order, resolution
schedule, and step budget. Record calibrated train/heldout metrics, optical
zoom/shift PSNR, physical stress, multilayer metrics, loss decomposition,
steps/s, GPU phase timing, and memory plan. Alternate run order and reject
contended host artifacts.

If depth-only variants slow non-depth steps materially, specialize the source
geometry checkpoint path by cadence. If reference projection is material,
compile a raster-only reference projector that omits the unused VJP packet.

