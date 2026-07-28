# Bounded browser dynamic 3DGS module

Added an isolated correctness-first dynamic-3DGS browser baseline without
wiring it into the SPA or Python trainer hierarchy.

## Contract

- Explicit `(frame, splat)` state, so storage and work scale with frame count.
- Calibrated pinhole projection of world-space anisotropic covariance.
- Stable ascending camera-space depth order per sampled ray.
- Front-to-back alpha compositing over black.
- Per-frame RGB-logit and opacity-logit Adam updates.
- Fixed means, log scales, and quaternions in this first bounded subset.

The restricted optimizer surface is deliberate. Geometry optimization would
require the full projection/covariance VJP; omitting it while moving means would
not be an honest dynamic-3DGS backward.

## Verification

- Node contract suite: 16 tests passed, including central finite differences
  for every optimized parameter of the new module.
- Browser validation scopes caught and fixed WGSL parser/pipeline errors before
  timing.
- Coffee Martini motion-ray benchmark, 8 frames, 2 train cameras, 16 splats,
  64 sampled rays/step, 200 timed steps:
  - 12,266 steps/s
  - 785,035 sampled pixels/s
  - 12,560,564 primitive evaluations/s
  - final sampled loss 0.105025
  - explicit state 8,192 bytes

This is not fast-mac parity: it has no tiling/binning, global sort, early exit,
SH, densification, geometry VJP, SSIM loss, or temporal regularization.
