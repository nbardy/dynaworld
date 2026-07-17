# Browser Adam And Density Replan

## Context

The browser trainer had a useful live WebGPU loop and honest source-view
metrics, but fixed splats and SGD-style updates were being tuned without the
optimizer/density-control machinery used by 3DGS-family systems.

## Current Model

The browser bottleneck is now primarily representation and work organization,
not a missing scalar knob. The train kernel evaluates all splats for every
sample independently for every splat, so adding windowed SSIM in-place would
multiply the dominant all-pairs work. Fixed-cap recycling is appropriate for
WebGPU because it changes support without reallocating buffers.

## Implemented

- Added first/second Adam moments for all 12 parameters per splat.
- Added persistent EMAs for absolute center gradient, contribution, opacity
  gradient, and motion gradient.
- Added fixed-cap maintenance every 256 steps: rank weak slots, rank sampled
  motion residuals, replace up to eight slots, and reset both ping-pong params,
  moments, and statistics for those slots.
- Added `Recycled` and `Param Delta` diagnostics.
- Serialized parameter readbacks and paused training during CPU validation.
- Guarded mode/splat resets until dataset loading is complete.

## Backtracks

Status: rejected.

The first Adam mapping reused SGD-scale rates. At 320 splats it raised motion
coverage but worsened grid loss from `0.000369` to `0.000400`. A direct
parameter-delta probe later showed `5.88e-2` mean absolute movement by step 526,
confirming over-large updates rather than a dead optimizer. Parameter groups
were reduced 10x.

Status: weakened.

Broad births at radius `0.018` increased coverage but caused an immediate loss
dip. Final births are localized at radius `0.010`, limited to eight, and seeded
from the highest sampled motion residuals. This mechanism is browser-green but
does not yet have a matched quality win.

Status: invalidated.

Several apparently unchanged metric traces were stale. Concurrent validation
and density maintenance mapped the same readback buffer, and forced validation
returned early while another read was busy. Serialized readbacks and explicit
validation-time train pauses are now part of the app contract.

## Assumptions And Boundaries

- The training objective remains the 128x128x8 source crop.
- The second preview camera is visual context, not supervision.
- The motion models remain simplified 2D screen-space World Tubes-style and
  linear dynamic-splat branches, not full Metal shader-family parity.
- Global-luma SSIM is validation-only; it is not windowed D-SSIM training.

## Falsification Tests

1. Build a tiled image-space loss/backward path and measure whether 3x3
   windowed D-SSIM costs less than 2x the current step at 320 splats.
2. Run matched `converge47` and Adam+density arms with identical init, samples,
   splats, steps, and validation cadence. Reject Adam+density unless motion loss
   improves without lower motion coverage or worse static coverage.
3. Add calibrated multicam supervision before using target-camera PSNR/SSIM as
   evidence. Source-view improvement alone cannot prove dynamic 3D recovery.

## Decision

Keep the Adam/statistics/fixed-cap machinery as the next experimental base, but
do not claim convergence. The next architecture task is image/tile-oriented
training or exported geometry initialization, not another support/LR sweep.
