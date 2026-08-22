# Browser StableGS headless compile and deploy

## What we tried

- Committed and pushed the fused Mip/dual-opacity/geometry-color/paired-depth
  browser stack on `codex/repository-audit-cleanup`.
- Triggered the checked-in Pages workflow. GitHub rejected it before checkout
  because the `github-pages` environment permits `main`, while repository
  Pages is actually configured in legacy mode from `gh-pages`.
- Published the browser subtree directly as a fast-forward `gh-pages` commit,
  preserving the established public-root layout and leaving `main` untouched.
- Installed Puppeteer only under `/private/tmp` and ran the existing headless
  Chrome/Dawn geometry experiment against local Chrome.

## Runtime failures that changed the code

The first headless attempt failed WGSL validation because checkpoint-block
backward generated an unused compact projection-VJP function against
`RasterProjection`. That hot raster packet intentionally omits
`cameraPointValid`; the real compact VJP happens later in the update module.
The fix keeps the unused helper on the complete `Projection` type and leaves
the 32-byte raster record unchanged. The ordinary staged pair path now also
uses `conicDepthAlpha.y` for raster depth rather than a missing VJP field.

The second attempt failed because `referenceDepth` declared
`array<atomic<u32>>` as `var<storage,read>`. WGSL requires storage atomics to be
`read_write` even when the entry point only loads them. Changing the access
mode fixed compilation without adding a write or synchronization operation.

The third attempt compiled every module used by the exact control and selected
full candidate, then completed training measurements.

## Diagnostic result

Workload: Coffee Martini calibrated train17/holdout1, 96x72, 8,192 splats,
RGBA8 target page, packed-FP16 checkpoints, split-compact projection,
checkpoint-block backward, 16 warmup and 64 measured steps.

- exact fast baseline: 499.2 steps/s;
- Mip 2D + dual opacity + geometry color 0.1 + paired depth every 8: 476.9
  steps/s;
- observed wall-throughput ratio: 0.955, or 4.5% slower;
- candidate round CV: 0.042;
- baseline round CV: 0.195;
- both finite, zero tile overflow, zero projection-VJP FP16 saturation.

This is not promotion evidence. Preflight found about 70% existing Apple GPU
utilization, load per logical CPU around 1.7, competing CPU fraction above
0.7, and swap around 0.31 of physical memory. The reversed-order run was
correctly blocked after this resource gate tripped. The completed artifact is
temporary at `/private/tmp/dynaworld-stablegs-geometry-headless.json`.

## What the result does not say

- It does not show improved PSNR, SSIM, heldout quality, glass quality, or
  camera-stress quality.
- The candidate objective includes geometry terms, so its scalar loss is not
  directly comparable to the baseline scalar loss.
- It does not establish a reliable 4.5% production overhead because the host
  was contended and the reversed-order mate did not run.
- It does establish that the selected fused path compiles, trains without the
  measured safety failures, and is in the same broad throughput regime as the
  exact control.

## Next experiment

Pause all interactive WebGPU trainers, wait for host GPU/load/swap gates to
clear, then run matched control-first and candidate-first artifacts. Quality
still needs A0-A5 training runs with heldout PSNR/SSIM plus optical and physical
camera-stress diagnostics. Only verifier-accepted retained artifacts should
update `BASELINES.md`.
