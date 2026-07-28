# DynaWorld Browser Trainer

A standalone WebGPU SPA for testing a compact dynamic Gaussian-splat trainer in
the browser. This directory is a demo and systems prototype. It is deliberately
separate from the Python paper-trainer hierarchy.

## Run

The worker runtime uses `SharedArrayBuffer`, so serve the repository with the
included cross-origin isolation headers:

```bash
python3 web/dynaworld_browser_trainer/serve_isolated.py --port 8080
```

Open:

```text
http://127.0.0.1:8080/web/dynaworld_browser_trainer/
```

Useful diagnostics:

```text
http://127.0.0.1:8080/web/dynaworld_browser_trainer/benchmarkTrainerBackends.html
http://127.0.0.1:8080/web/dynaworld_browser_trainer/benchmarkLegacy2d.html
http://127.0.0.1:8080/web/dynaworld_browser_trainer/workerSmoke.html
```

Run the browser unit suite:

```bash
cd web/dynaworld_browser_trainer
npm test
```

## Current Data Contract

The default checked-in bundle is
`coffee_martini_train17_holdout1.json`:

| Field | Current value |
| --- | --- |
| Scene | Neural 3D Video Coffee Martini |
| Raster | 96x72 |
| Times | 16 synchronized frames sampled across the 300-frame source |
| Training cameras | 17 |
| Heldout cameras | `cam06` only |
| Initialization | 4,096 train-visible Ex4DGS SfM XYZRGB points |
| Anchor coordinates | `cam04` OpenCV camera frame |

The browser does not run COLMAP. The bundle exporter is a thin adapter over
`src/train/multicam_video_data.py`; it preserves the canonical manifest,
camera calibration, train/heldout split, synchronized frame indices, and
initialization provenance. Heldout pixels are never used by the optimizer.

Regenerate a bundle with `src/train/export_dynaworld_browser_bundle.py` and the
checked-in manifest. The adapter test is
`tests/test_browser_multicam_export_adapter.py`.

## Active Training Backends

The SPA exposes two production selectors:

1. `tiled3d`, the default full-frame path.
2. `sampled3d`, the older sampled-ray control.

They share the same 24-float primitive schema, initialization, cameras, render
path, worker protocol, and UI. They do not share an objective, so raw steps per
second are not a quality comparison.

`trainerWebGpuStar.js` and `trainerWebGpuDynamicGs.js` are bounded,
correctness-first probes. They are not hidden SPA backends:

- the STAR probe is an affine camera-space shared-adjoint fixture;
- the DynamicGs probe has independent frame state and only a partial optimized
  parameter set.

Neither may be presented as a faithful World Tubes, native 4DGS, or complete
dynamic 3DGS implementation.

## Tiled Full-Frame Step

One optimizer step selects one training camera and one time. A deterministic
coprime permutation visits every camera/time pair before repeating.

The GPU command buffer then performs:

1. clear tile counts, metrics, and pair bookkeeping;
2. project every splat through the calibrated pinhole camera;
3. bin exact opacity-aware ellipse support into 16x16 tiles;
4. depth-sort each tile;
5. front-to-back alpha compositing with periodic transmittance checkpoints;
6. local SSIM statistics and image-space RGB gradient;
7. pair-owned shared raster backward;
8. per-splat reduction and Adam update;
9. optional fixed-capacity split/recycle maintenance.

This is a real full-raster training step. It does not evaluate every splat for
every image pixel, and it does not materialize a pixel-by-splat gradient tensor.

The default objective is:

```text
0.8 * mean(abs(prediction - target)) + 0.2 * mean(1 - SSIM)
```

The current SSIM uses an 11x11 reflected **uniform box** window. Radius 5 and
the constants `C1=0.01^2`, `C2=0.03^2` match common SSIM settings, and the
analytic CPU analogue passes finite differences. However, canonical SSIM and
the official 3DGS implementation use an 11x11 Gaussian window with sigma 1.5.
This distinction must be preserved in baseline claims.

## Primitive Model

Each splat stores:

- base 3D center and a trainable static/dynamic mix;
- linear 3D velocity and temporal center;
- optional sinusoidal 3D center offset;
- three world-space log scales;
- normalized quaternion rotation;
- constant RGB and opacity logit.

The covariance is projected through the camera Jacobian. The backward includes
center, perspective depth, scale, quaternion, color, opacity, temporal-center,
velocity, harmonic-offset, and static-mix derivatives. Scale aspect ratio is
bounded to 3:1.

The `Motion Model` selector is only a trajectory-basis ablation:

- `Harmonic trajectory splats` adds one sinusoidal center component.
- `Linear trajectory splats` omits it.

Neither mode uses a native four-dimensional covariance.

## Initialization And Topology

The exporter transforms the external SfM point cloud into the declared anchor
camera, keeps points visible from at least one training camera, and takes a
deterministic farthest-point subset. Initial scale and rotation come from local
point-cloud neighborhoods; opacity starts at 0.1.

The trainer uses fixed GPU capacity:

- when requested splats are below capacity, hidden slots are filled by
  split/recycle events beginning at step 600;
- after fill, weak slots are recycled from high-gradient parents every 500
  steps through step 60,000;
- at the default 4,096 splats, capacity is already full, so maintenance
  recycles rather than increasing primitive count.

This is dynamic topology within a fixed allocation. It supports relocation,
spatial scale shrinkage, opacity-mass preservation, temporal separation, and
moment reset. It is not a dynamic buffer resize, canonical 3DGS densification,
or native 4DGS spatial-temporal pruning schedule.

## Runtime And Metrics

Training runs in a dedicated worker. It submits bounded eight-step bursts and
keeps no more than 32 GPU steps queued. Queue completion probes publish actual
completed throughput rather than command-submission speed.

Live rendering is capped at 20 GPU frames per second and shows two training
cameras plus the heldout camera at a looping time. It can be disabled without
stopping optimization.

Loss readback runs every 256 steps. Full metrics request an asynchronous
parameter snapshot, then a separate validation worker computes a deterministic
12x12 grid across all train cameras and the heldout camera every 2,048 steps.
The charts retain full decimated history and show:

- optimizer objective;
- sparse train and heldout MSE/MAE/PSNR;
- sparse global-luma SSIM proxy.

The validation SSIM is not the training windowed SSIM and is not a
paper-protocol full-image metric.

## July 28 Scaling Result

The synchronized Apple M4 benchmark creates a fresh WebGPU device per repeat,
uses matched requested splats and capacity, warms for 32 steps, measures three
GPU-drained intervals, and excludes compilation, initialization, preview,
validation, and readback.

| Raster | 768 splats | 1,536 splats | 4,096 splats |
| --- | ---: | ---: | ---: |
| 96x72 | 1,268 steps/s | 933 steps/s | 470 steps/s |
| 192x144 | 699 steps/s | 557 steps/s | 386 steps/s |

Four times as many pixels retain 55%, 60%, and 82% of native-raster step rate
as splat count rises. In this range, splat-side work is the stronger cost.

A matched 96x72/1,536-splat window ablation measured:

| SSIM window | Median steps/s |
| --- | ---: |
| 7x7 box | 1,174 |
| 11x11 box | 933 |

The smaller window is about 26% faster, but changes the objective and has no
quality result. The standards-preserving optimization is an 11-tap separable
Gaussian forward and transpose backward, not a whole-image SSIM statistic.

The 384x288/768-splat probe fails the current resource contract. The complete
18-camera x 16-time RGBA32F target tensor becomes about 486 MiB in one storage
binding. Higher-resolution work therefore needs streamed or paged camera/time
targets before more shader tuning.

Full measurements and repeats are in
`benchmark_results/2026-07-28_tiled_scaling_apple_m4.json`.

## What The Numbers Do Not Prove

- The live SPA is slower than the isolated table because preview, worker
  scheduling, status publication, metric readback, and validation coexist with
  training. A recent 4,096-splat observation was about 250-280 completed
  steps/s.
- Historical 7.3, 584-824, and 793 steps/s observations belong to different
  sampled-ray kernels, splat counts, objectives, and UI schedules.
- No saved Metal measurement matches this complete browser step. Projection,
  raster-only, or synthetic Metal microbenchmarks cannot establish a
  WebGPU-versus-Metal percentage.
- Throughput does not establish convergence or novel-view quality.

## Remaining Baseline Work

The browser trainer is now a useful full-frame systems prototype, but not a
solid research baseline. The highest-value missing evidence is:

1. CPU-versus-WGSL rendered-image and gradient parity on a tiny tiled fixture.
2. Canonical Gaussian-window SSIM with value/gradient parity.
3. Phase timing for bin/sort, raster, SSIM, backward, update, preview, and
   validation.
4. Matched fixed-topology versus split/recycle quality runs.
5. Matched initialization and splat-capacity ablations.
6. Full-image heldout PSNR, SSIM, LPIPS, and L1 on more than one scene and seed.
7. A complete calibrated dynamic-3DGS baseline before promoting native 4DGS or
   World Tubes to a selectable browser backend.

See `research_notes/browser_4dgs_baseline.md` for the external native-4DGS
comparison contract.
