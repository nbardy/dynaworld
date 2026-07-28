# Browser depth, density, and native 4DGS baseline pass

## Scope

Fresh-state audit and implementation pass after the calibrated browser trainer
showed learned anisotropy but plateaued without recognizable structure.

## Findings

- The remembered early high-quality browser result was the legacy single-view
  screen-space trainer, initialized from target pixels and rendered over the
  target temporal mean. It was not a lost calibrated 3D result.
- The active harmonic and linear options are trajectory bases in one anisotropic
  3D Gaussian kernel. Neither is native World Tubes, native 4DGS, or canonical
  Dynamic 3DGS.
- The standalone STAR and DynamicGs shaders are microbenchmarks: STAR is a
  32-tube affine single-camera lane; DynamicGs is a 32-splat depth-order oracle
  whose geometry does not receive gradients. They are not honest SPA backends.
- The old density mechanism existed only in the 2D trainer. It recycled weak
  slots onto high-residual target pixels through CPU readback; it did not clone
  and split 3D Gaussians.

## Changes

- Added a GPU camera/time order-cache pass for each active train camera and
  frame. Sampled-ray compositing reads its far-to-near primitive permutation.
- Added a GPU bitonic depth sort for each live preview panel and matching CPU
  validation ordering.
- Made the per-splat static/dynamic mixture trainable with its analytic gate
  derivative.
- Added a GPU-only fixed-capacity maintenance pass every 512 steps through
  16,384. It selects four weak victims and four useful parents, splits along the
  largest covariance axis, conserves combined opacity mass, optionally separates
  temporal centers, and resets relevant Adam/statistics state.
- Added a native 4DGS baseline contract at
  `research_notes/browser_4dgs_baseline.md`. The paper is a relevant Coffee
  Martini reference but has no matched repo result yet.

## Verification

- `node --test web/dynaworld_browser_trainer/tests/*.test.mjs`: 26 passing.
- Live Apple WebGPU initialization compiled all new WGSL pipelines without
  console warnings or errors.
- The 768-splat train17/holdout1 run sustained about 853 completed steps/s early
  and 774.5 steps/s in the long trace. At paused step 180,076 it reported loss
  0.01014, heldout PSNR 13.3 dB / SSIM proxy 0.427, mean aspect 1.39:1, and
  exactly 128 split/recycle operations before the configured density stop.
- Training, sorting, rendering, validation snapshots, and maintenance remain
  asynchronous; no density or order readback was introduced into the pump.

## Remaining Gates

- This pass fixes ordering and fixed-capacity topology evolution but does not
  prove a quality promotion. Run matched fixed-seed/no-maintenance versus
  maintenance and fixed-order versus depth-order ablations at equal wall time.
- Native 4DGS still requires a real 4D covariance/conditioning renderer and
  matched official baseline adapter. Do not rename harmonic trajectory splats.
- Windowed SSIM/DSSIM, view-dependent appearance, tiled active lists, and a
  larger useful primitive budget remain later gates.
