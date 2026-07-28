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
http://127.0.0.1:8080/web/dynaworld_browser_trainer/tiledParityHarness.html
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
| Checked-in initialization | 4,096 train-visible external Ex4DGS XYZRGB points |
| Anchor coordinates | `cam04` OpenCV camera frame |

The browser does not run COLMAP. The bundle exporter is a thin adapter over
`src/train/multicam_video_data.py`; it preserves the canonical manifest,
camera calibration, train/heldout split, synchronized frame indices, and
initialization provenance. Heldout pixels are never used by the optimizer.

The checked-in bundle predates the provenance report contract. Its external
Ex4DGS cloud is filtered after loading to points visible from at least one
training camera, but the repository cannot prove which source images originally
created that cloud. In particular, it cannot prove that `cam06` was excluded.
The SPA therefore labels this initialization `unverified`; "train-visible" is
not the same claim as "constructed from train cameras only."

### Verified offline initialization

Coffee Martini already provides calibrated intrinsics and camera poses. The
missing operation is known-pose feature triangulation, not a new browser
structure-from-motion system. Run the existing pycolmap adapter offline on one
synchronized frame from the 17 train cameras:

```bash
PYTHONPATH=src/train uv run --with pycolmap==4.0.4 python \
  research_experiments/dynamic_foam/build_pycolmap_known_pose_point_cloud.py \
  src/train_configs/browser_coffee_martini_train17_known_pose_sfm.jsonc \
  --output research_experiments/dynamic_foam/artifacts/browser_coffee_martini_train17_known_pose_frame0_1024px.ply \
  --target-size 1024 \
  --frame-index 0 \
  --camera-model pinhole \
  --camera-mode per_image \
  --feature-backend pycolmap \
  --feature-type sift \
  --matcher-type sift_bruteforce \
  --pairing-neighbors 4 \
  --known-pose-guided-verification \
  --min-track-length 2 \
  --min-unique-cameras 2 \
  --max-points 16384
```

The builder writes both the PLY and a same-stem JSON report. The report declares
the input cameras, train-only verification, and `model` coordinate frame. The
2026-07-28 run produced 815 bounded points from 71,890 keypoints and 38 verified
camera pairs. That is a valid sparse scaffold, but it does not satisfy the
current 4,096-seed bundle. A valid 768-seed ablation export is:

```bash
PYTHONPATH=src/train uv run python src/train/export_dynaworld_browser_bundle.py \
  --dataset-manifest src/dataset_configs/neural3d_coffee_martini_train17_holdout1_full_300f_manifest.jsonl \
  --dataset-sample-id neural3d_coffee_martini_train17_holdout_cam06_full_300f \
  --dataset-split train17_holdout1 \
  --dataset-output web/dynaworld_browser_trainer/coffee_martini_train17_holdout1_verified_sparse.json \
  --seed-point-cloud research_experiments/dynamic_foam/artifacts/browser_coffee_martini_train17_known_pose_frame0_1024px.ply \
  --seed-provenance-report research_experiments/dynamic_foam/artifacts/browser_coffee_martini_train17_known_pose_frame0_1024px.json \
  --dataset-height 72 \
  --dataset-width 96 \
  --dataset-frame-count 16 \
  --dataset-native-frame-count 300 \
  --dataset-seed-count 768
```

The exporter rejects heldout camera overlap, unknown coordinate frames,
unverified reports without an explicit override, and point clouds with fewer
than the requested number of train-visible points. This also prevents the old
browser fallback from silently duplicating a sparse seed cloud. Do not replace
the checked-in default with the 768-point variant until it has a matched
initialization/densification quality run.

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
5. front-to-back alpha compositing with storage-bounded transmittance checkpoints;
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

The default SSIM uses the standard 11x11 Gaussian window with sigma 1.5,
reflected image boundaries, and `C1=0.01^2`, `C2=0.03^2`. Its analytic image
gradient passes finite differences, and the active Apple WebGPU forward,
objective, and selected parameter-family gradients pass the live tiled parity
harness. Nondefault SSIM radii remain normalized uniform-window probes and
change the objective.

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

Loss readback runs every 256 steps. Every 8,192 steps, full metrics request an
asynchronous parameter snapshot and send it to a separate validation worker.
The optimizer pump continues submitting work while the GPU copy/map completes;
the full-image raster and metric pass never runs in the training worker.

The validation selection is all pixels from `cam04` and `cam09` over all 16
times, plus all pixels from heldout `cam06` over all 16 times. The charts retain
full decimated history and show:

- optimizer objective;
- full-image train and heldout MSE/MAE/PSNR;
- full-image channelwise 11x11 Gaussian SSIM;
- center, motion, scale, rotation, color, and opacity RMS deltas between
  validation snapshots.

This is a deterministic browser diagnostic, not the Python paper evaluator:
it currently omits LPIPS and evaluates two representative train cameras rather
than all 17. A measured 4,096-splat validation pass took about 4.5 seconds on
the local Apple browser in the earlier contended run and 1.4-1.9 seconds in the
final clean smoke, without pausing the optimizer.

## July 28 Scaling Result

The synchronized Apple M4 benchmark creates a fresh WebGPU device per run, uses
matched requested splats and capacity, warms for 32 steps, measures 128
GPU-drained steps, and excludes compilation, initialization, preview,
validation, and readback. The table uses FP32 checkpoints; the 384x288/4,096
endpoint is the median of three repeats and the other cells are single
intervals.

| Raster | 768 splats | 1,536 splats | 4,096 splats |
| --- | ---: | ---: | ---: |
| 96x72 | 1,233 steps/s | 863 steps/s | 359 steps/s |
| 192x144 | 675 steps/s | 462 steps/s | 240 steps/s |
| 384x288 | 266 steps/s | 182 steps/s | 118 steps/s |

At 4,096 splats, packed-FP16 checkpoints improve these three raster points to
403, 294, and 132 steps/s respectively.

A matched 96x72/1,536-splat window ablation measured:

| SSIM window | Median steps/s |
| --- | ---: |
| 7x7 box | 1,174 |
| 11x11 box | 933 |

The smaller window is about 26% faster, but changes the objective and has no
quality result. The standards-preserving optimization is an 11-tap separable
Gaussian forward and transpose backward, not a whole-image SSIM statistic.

These saved scaling artifacts predate the switch from the 11x11 box window to
the 11x11 Gaussian weights. They have the same 121-sample support but are not
an exact throughput claim for the current build until rerun.

### Memory-layout follow-up

The primary tiled backend now pages exactly one camera/time RGBA32F target into
a reusable GPU buffer before each submitted step. At 384x288 this is 1.69 MiB,
down from a 486 MiB all-camera/all-time binding, without changing target
precision or the objective. Queue writes and compute submissions remain ordered
on one `GPUQueue`; training does not drain or synchronize between pages.

Each tile now reserves enough IDs to cover every splat, so a valid frame cannot
overflow its bin. Active tile/splat pairs are compacted, backward dispatch uses
two workgroup dimensions, and pair-owned gradients accumulate into one FP32
record per splat with compare/exchange atomics. Checkpoint stride expands only
enough to keep each storage binding within the device limit. On the Apple M4's
128 MiB binding limit, the 384x288/4,096-splat plan is:

| Buffer | Old reservation | Current reservation |
| --- | ---: | ---: |
| Target | 486 MiB | 1.69 MiB |
| Forward checkpoints | 432 MiB | 108 MiB |
| Pair gradients | 162 MiB | removed |
| Compact pair IDs and references | 13.5 MiB | 13.5 MiB |
| FP32 gradient accumulator | n/a | 0.375 MiB |

A real 384x288/4,096-splat browser smoke compiled, trained, reported finite
loss, and had zero tile overflow. FP32 repeats measured 117.6, 118.0, and 118.7
steps/s. Packed-FP16 repeats measured 131.1, 131.6, and 139.2 steps/s, an 11.6%
median improvement. After 1,024 submissions, FP16 and FP32 losses were
0.284516 and 0.284515; the corresponding measured intervals were 100.0 and
88.1 steps/s.

Packed-FP16 checkpoints are therefore the SPA default, with FP32 selectable.
Packing uses core `pack2x16float`/`unpack2x16float`; it does not require native
FP16 arithmetic. Projection, covariance, depth, compositing arithmetic, SSIM,
image gradients, atomic reductions, trainable parameters, and Adam moments
remain FP32. FP16 either halves checkpoint memory or spends that saving on
denser checkpoints and less backward replay, depending on raster size.

The sampled-ray control still binds the complete target tensor and therefore
still rejects 384x288. More importantly, GPU paging does not remove the host
Float32 tensor: a future high-resolution SPA path should retain canonical RGBA8
frames and share them across the main, training, and validation workers instead
of cloning roughly 486 MiB per worker.

Full measurements and repeats are in
`benchmark_results/2026-07-28_tiled_scaling_apple_m4.json` and
`benchmark_results/2026-07-28_tiled_memory_precision_apple_m4.json`.

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

## Baseline Status

The browser trainer is a useful full-frame systems prototype, but not yet a
solid research baseline. Completed correctness gates now include:

1. active Apple WebGPU image/objective/selected-gradient parity on an eight-splat
   tiled fixture;
2. canonical 11x11 Gaussian SSIM value and image-gradient finite differences;
3. deterministic full-image train/heldout MSE, MAE, PSNR, and SSIM;
4. fail-closed initialization provenance and coordinate-frame handling.

The highest-value remaining evidence is:

1. compare the verified 768-point train-only cloud plus growth against the
   legacy unverified 4,096-point seed under matched settings, or produce a
   denser verified cloud with a stronger matcher;
2. phase timing for bin/sort, raster, SSIM, backward, update, preview, and
   validation;
3. matched fixed-topology versus split/recycle quality runs;
4. matched initialization and splat-capacity ablations;
5. full-image heldout PSNR, SSIM, LPIPS, and L1 on more than one scene and seed;
6. a complete calibrated dynamic-3DGS baseline before promoting native 4DGS or
   World Tubes to a selectable browser backend;
7. canonical byte targets plus shared host storage before presenting 384x288 as
   a normal SPA dataset mode rather than an isolated GPU benchmark.

See `research_notes/browser_4dgs_baseline.md` for the external native-4DGS
comparison contract.
