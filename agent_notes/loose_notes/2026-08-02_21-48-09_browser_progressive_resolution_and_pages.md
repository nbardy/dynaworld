# Browser progressive resolution and static deployment

## Goal

Make the browser trainer edge-to-edge, preserve the fast 96x72 convergence
phase, continue the same optimization at native 384x288, and publish a static
build without reintroducing main-thread or validation stalls into the train
loop.

## Design

- Default resolution mode is `progressive-96-384`.
- Coarse training runs through step 8,192. This is late enough to establish
  geometry while leaving reserved topology capacity for high-resolution
  residuals; switching near topology saturation would make the fine stage much
  less useful.
- The 384x288 RGBA8 target bank preloads while the worker continues coarse
  training.
- The worker performs one explicit transition. It stops new submissions,
  drains already-submitted GPU work, snapshots bounded continuation state,
  rebuilds resolution-dependent resources, restores state, restarts validation,
  and resumes if it was running.
- Continuation state includes both ping-pong parameter buffers, both Adam
  moments, density statistics, initial parameters, step, current buffer index,
  topology count, active prefix, recycle count, and cumulative tile diagnostics.
- Compatibility gates cover calibration, camera ordering, splits, frames, seed
  geometry, parameter schema, capacity, geometry scale, and density schedule.
- Loss/PSNR/SSIM history remains complete and marks the resolution boundary
  with a cyan vertical line.

This creates one bounded pause at a meaningful objective change. It does not
put evaluation, readback, or UI work into the steady-state optimizer pump.

## UI and hosting

- Removed outer gutters, rounded framing, and card padding from the comparison,
  metric, control, stats, and status bands.
- Kept the 3x2 GT/result matrix as the first task surface and metrics immediately
  below it.
- Added a GitHub Pages workflow that deploys only
  `web/dynaworld_browser_trainer`.
- Added a same-origin service worker that injects COOP/COEP on static hosts. A
  first hosted visit reloads once; later visits can use SharedArrayBuffer rather
  than clone the high-resolution target bank into both workers.
- The repository Actions job could not start because the GitHub account was
  locked by a billing issue. A static subtree commit was pushed to `gh-pages`
  instead, and the legacy Pages build completed successfully at
  `https://nbardy.github.io/dynaworld/`.

## Measurements

The existing matched Apple M4 live smoke remains the throughput evidence:

| Native resolution | Completed steps/s | GPU buffers |
| --- | ---: | ---: |
| 96x72 | about 309 | 11.3 MiB |
| 384x288 | about 130 | 97.6 MiB |

Sixteen times the pixels cost about 2.4x throughput in that configuration;
projection, sorting, optimizer work, and scheduling do not all scale with image
area.

The 2026-08-02 live transition smoke crossed at step 8,200 because the worker
submits eight-step bursts. It preserved the visible fit and finite objective,
reported 97.3 MiB, and continued past step 10,472. The live UI reported about
80 steps/s at 384x288 while tests, browser automation, and other development
jobs shared the machine. Treat that as a continuity result, not a new speed
baseline. The separate high-resolution validation later reported 22.1 dB /
0.750 SSIM on the two representative train cameras and 23.2 dB / 0.733 SSIM
on the heldout camera. That CPU validation took 67.6 seconds and further contaminated the live
throughput observation without synchronizing the training worker.

## Floater audit

The camera-axis correction remains strongly supported: median epipolar error
fell from roughly 60 px to 0.48 px and a prior fresh `cam06` run reached 27.0 dB
/ 0.909 SSIM. Current orbit views can leave the calibrated camera hull and must
not be conflated with heldout-camera quality.

The highest-probability remaining cause is fixed-budget allocation. Splits use
screen gradient, alpha, and velocity gradient, but not pixel residual, depth,
per-view contribution, or multi-view support. Once capacity is full there is no
relocation of low-value opaque splats. The next narrow lane is therefore:

1. deterministic orbit alpha/depth traces;
2. residual-weighted per-splat contribution over the camera cycle;
3. fixed-budget relocation from low-contribution slots to high-error pixels
   with multi-view support;
4. only then, diagnostic-gated local rigidity or improved sorting.

SpacetimeGS error/coarse-depth guided births, 3DGS-MCMC relocation, and revised
pixel-error densification are the closest paper precedents. Sparse-view dropout
is not a justified default for a 17-train-camera dynamic sequence.

## Verification

- All 159 browser trainer tests passed after adding the worker handoff contract.
- JavaScript syntax checks and `git diff --check` passed.
- Live WebGPU run completed the 96x72 to 384x288 handoff without resetting the
  step or model, continued training, rendered all three views, and emitted no
  browser warnings/errors.
- Desktop and compact viewport checks showed no horizontal overflow or element
  overlap. Canvas content remained nonblank after the transition.
- The public build completed its first-visit service-worker reload, reported
  `atomic SAB`, initialized the Apple WebGPU tiled backend, and rendered all
  three result views.

## Scope discipline

This remains a browser demo adapter. It reuses the checked-in calibrated bundle
contract and does not add a parallel Python trainer, split convention, camera
stack, or research-model lane. Unrelated paper-protocol and World Tubes changes
already present in the dirty worktree were not touched or staged.
