# Browser Baseline Gates And Known-Pose Initialization

Date: 2026-07-28

## Scope

Keep `web/dynaworld_browser_trainer/` a calibrated demo and systems prototype.
Do not add another Python trainer hierarchy, calibration contract, or research
representation lane. Reuse the canonical multicamera manifest, split, poses,
and existing offline geometry builder.

## Initialization Audit

- The checked-in `coffee_martini_train17_holdout1.json` starts from 4,096
  external Ex4DGS XYZRGB points in the cam04 anchor frame.
- The exporter filters loaded points to those visible from at least one train
  camera and takes a deterministic farthest-point subset.
- The browser normalizes scene/camera translations by median positive anchor
  depth. It initializes each center and RGB from a seed point.
- Local eight-neighbor covariance supplies anisotropic scales and a quaternion;
  aspect ratio is bounded to 3:1.
- Static mix starts at `0.92`, velocity and harmonic motion at zero, temporal
  center at `0.5`, and opacity at `0.1`.
- This is not random initialization and it is not pixel-perfect
  initialization. Points/colors are sparse multiview geometry; every parameter
  must still be optimized against images.
- The external Ex4DGS cloud does not record its input cameras. Post-load
  train-visibility filtering cannot prove cam06 was absent during original
  reconstruction. It must therefore be labelled `unverified`.

The correct replacement is the existing known-pose pycolmap builder. Neural3D
already supplies calibrated intrinsics and extrinsics, so rebuilding camera
poses in WebGPU would duplicate semantics and add a large, low-value system.
Use one synchronized frame from all 17 train cameras to triangulate static scene
support while excluding heldout cam06. Additional times risk mixing moving
foreground observations into nominal static tracks.

## Implemented Gates

- Browser export now requires a provenance report unless unverified external
  use is explicitly opted into.
- Reports declare method, input cameras, train-only verification, and coordinate
  frame. Heldout overlap and non-train cameras fail closed.
- `world` seed points are transformed once to the anchor frame. `model` and
  explicit anchor-frame points are not transformed again. This fixes a
  potential double-transform for known-pose reconstructions.
- Export fails when fewer train-visible points exist than requested instead of
  leaving the browser to duplicate sparse seeds.
- The known-pose builder now emits a report directly consumable by the exporter.
- A checked-in Coffee Martini config pins the canonical train17/holdout-cam06
  split and one-frame geometry construction.
- The first executable 1024px run exposed that exhaustive SIFT brute-force
  matching spends most of its time on all 136 camera pairs despite known poses.
  A default-off `--pairing-neighbors` option now writes an imported pycolmap
  pair list from nearest camera centers at each synchronized time. Existing
  experiments retain exhaustive behavior; the browser recipe uses four nearest
  cameras and never creates cross-time pairs.
- The real train17 run completed in about 2.5 minutes: 71,890 keypoints, 38
  matched/verified image pairs, 881 raw points, and 815 bounded points with
  mean/median reprojection error `1.38/1.24` pixels. All input cameras are train
  cameras and cam06 is held out.
- This is below the 4,096-point default. The exporter rejects that request
  before decoding videos. A denser all-pairs/no-cross-check SIFT attempt was
  stopped after more than ten minutes in matching; it was not a useful baseline
  trade. The next admissible comparison is 768 verified seeds plus measured
  growth versus the legacy 4,096 external seeds, or a stronger matcher.
- Orthographic inspection shows coherent near-room surface fragments around
  model depth 5-15, but also sparse disconnected distant clusters out to depth
  about 85. This is plausible sparse geometry, not a dense scene scaffold and
  not a drop-in quality promotion.

## Training And Validation

- Default tiled training now uses `0.8 L1 + 0.2 (1 - SSIM)` with the standard
  reflected 11x11 Gaussian window, sigma 1.5.
- The CPU image objective and Gaussian SSIM gradient pass finite differences.
- The active Apple WebGPU parity harness uses the production shader at zero
  learning rate and compares rendered RGBA, objective terms, and selected
  parameter-family gradients against CPU/finite-difference references.
- Live parity passed with maximum RGB error `8.20e-8`, alpha error `2.16e-7`,
  objective-term error below `9e-8`, zero tile overflow, and seven active
  gradient families. Active relative gradient error was below `6e-5`.
- An initial center/velocity parity failure was a diagnostic finite-difference
  step crossing the opacity-support cutoff. Reducing the geometry perturbation
  restored local finite-difference stability; the shader gradient was correct.
- Validation now rasterizes every pixel for cam04/cam09 and heldout cam06 over
  all 16 frames in a separate worker. It reports MSE, MAE, PSNR, channelwise
  Gaussian SSIM, coverage, and per-family parameter RMS deltas.
- The training pump continues submitting while the asynchronous parameter copy
  completes; full raster/metrics work does not execute on the training worker.
- Live 4,096-splat validation passes ranged from 1.4 to 4.5 seconds depending
  on host contention, so automatic validation moved from every 2,048 to every
  8,192 steps.
- A live optimizer snapshot showed nonzero center, motion, scale, rotation,
  color, and opacity updates, ruling out a globally frozen optimizer.

## Verification

- Browser Node suite: `62 passed`.
- Focused exporter/known-pose pytest suite: `14 passed`.
- `node --check` and `git diff --check`: passed before documentation updates.
- Main Apple WebGPU SPA: initialized, trained, validated, and rendered without
  console warnings/errors.

## Remaining Evidence

- Visually inspect the generated 815-point train-only cloud and run a matched
  768-seed-plus-growth browser ablation.
- Run matched legacy-unverified versus verified-init quality.
- Run fixed-topology versus split/recycle and splat-capacity ablations.
- Add phase timing and LPIPS.
- Require multiple scenes and seeds before any `BASELINES.md` promotion.
- Do not expose the bounded STAR or DynamicGs probes as production backends
  until each has a complete calibrated model/objective/optimizer contract.
