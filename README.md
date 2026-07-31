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
http://127.0.0.1:8080/web/dynaworld_browser_trainer/benchmarkTiledKernels.html
http://127.0.0.1:8080/web/dynaworld_browser_trainer/benchmarkLegacy2d.html
http://127.0.0.1:8080/web/dynaworld_browser_trainer/tiledParityHarness.html
http://127.0.0.1:8080/web/dynaworld_browser_trainer/workerSmoke.html
```

The kernel benchmark does not require an interactive page. Bun owns a tiny
no-store server, launches a private headless Chrome/Dawn process, waits for the
WebGPU work, writes JSON, and closes both processes:

```bash
bun web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js \
  --experiment backward --variant both --splats 30000 \
  --warmup 32 --steps 128 --profiles 5 --tile-capacity 4096 \
  --out /tmp/dynaworld_backward_30k.json

bun web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js \
  --experiment projection --variant both --order candidate-first \
  --splats 30000 --warmup 32 --steps 128 --profiles 5 \
  --tile-capacity 4096

bun web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js \
  --experiment ssim --variant both --splats 8192 --scale 2 \
  --warmup 32 --steps 128 --profiles 5

bun web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js \
  --experiment backward --variant both --splats 8192 \
  --warmup 32 --steps 128 --profiles 5 \
  --out-dir web/dynaworld_browser_trainer/benchmark_results/runs \
  --run-id staged-backward-8k-control-first

bun web/dynaworld_browser_trainer/summarize_headless_kernel_pair.js \
  web/dynaworld_browser_trainer/benchmark_results/runs/2026-07-31/backward_8k_control_first_v3_apple_m4.json \
  web/dynaworld_browser_trainer/benchmark_results/runs/2026-07-31/backward_8k_candidate_first_v3_apple_m4.json
```

Bun itself does not expose `navigator.gpu`; the hidden browser is the WebGPU
runtime, not the benchmark UI. The HTML lab remains available as an optional
interactive microscope for inspecting individual variants.

New `v3` artifacts fail closed for performance promotion. A run is promotable
only when all variants have finite loss, zero tile overflow, at least two
measurement rounds, per-round throughput coefficient of variation at or below
`0.10`, and a quiet host preflight/postflight. The Bun runner records:

- CPU busy fraction over a timed sample;
- load per logical CPU and aggregate process pressure;
- macOS memory pressure, swap occupancy, and thermal warnings;
- Apple driver GPU/renderer/tiler utilization before Chrome starts;
- the same host checks after a ten-second owned-queue cooldown;
- sanitized top-process basenames and categories, never arguments or PIDs;
- per-round throughput, CV, first/last drift, and execution-position bias.

The default `--contention-policy warn` still writes a diagnostic artifact but
sets `validity.promotable=false`. Use `--contention-policy fail` when running a
canonical sweep so a busy machine is rejected before Chrome starts. A quiet
snapshot is necessary but not sufficient: constant-rate GPU contention can
evade round CV, so promoted comparisons still need alternating execution order
and a reversed-start repeat. See
[`benchmark_results/README.md`](benchmark_results/README.md) for artifact
layout and naming.

Swap occupancy is part of promotion validity, not just recorded metadata. The
default `--max-swap-used-fraction 0.90` rejects canonical timing runs on a
heavily paged host even when macOS still reports an acceptable free-memory
percentage. The independent `--max-swap-to-memory-fraction 0.25` ceiling also
prevents macOS growing the swap pool from making the same used bytes appear
healthy.

Before launching Chromium, the runner also derives the requested dataset,
per-variant WebGPU buffers, largest storage binding, and conservative unified
memory headroom. The plan is saved as `requestedResourcePlan`; its byte model
is regression-checked against the saved 30K/96 allocation artifacts plus
subsequent explicitly tested layout deltas. A
30K/384x288 packed-checkpoint run resolves to a 108 MiB largest binding and
requires about 2 GiB of estimated available host memory before browser startup.
Use `--preflight-only` to print this plan and host assessment without creating
a browser or WebGPU device.

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
| Pose convention | LLFF raw `[down, right, back]` to OpenCV `[right, down, forward]` (`v2`) |

The browser does not run COLMAP. The bundle exporter is a thin adapter over
`src/train/multicam_video_data.py`; it preserves the canonical manifest,
camera calibration, train/heldout split, synchronized frame indices, and
initialization provenance. Heldout pixels are never used by the optimizer.

The pose-source identifier is
`neural_3d_llff_opencv_relative_pinhole_v2`. Raw `poses_bounds.npy` rows store
camera basis columns as `[down, right, backwards]`; the loader reorders them to
OpenCV `[right, down, forwards]` before forming anchor-relative poses. The
superseded sign-only conversion is intentionally a different artifact identity.

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
camera pairs, but it used the superseded LLFF axis conversion. Retain that
artifact as failure evidence only and rerun the builder under the `v2` pose
source before using a sparse scaffold. Once rebuilt, a 768-seed ablation export
is:

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

The SPA exposes three selectors:

1. `tiled3d-fast`, the default full-frame path.
2. `tiled3d`, the direct-backward full-frame reference.
3. `sampled3d`, the older sampled-ray control.

They share the same 24-float primitive schema, initialization, cameras, render
path, worker protocol, and UI. The two tiled paths share the same full-image
objective; their one-step parity is checked directly. The sampled path does not
share that objective, so raw steps per second are not a quality comparison.

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
3. bin exact opacity-aware ellipse support into 8x8 tiles;
4. depth-sort each tile;
5. front-to-back alpha compositing with storage-bounded transmittance checkpoints;
6. exact separable local SSIM statistics and transpose image gradient;
7. checkpoint-block raster backward into compact projected gradients;
8. one projection/covariance VJP plus Adam update per splat;
9. optional fixed-capacity split/recycle maintenance.

This is a real full-raster training step. It does not evaluate every splat for
every image pixel, and it does not materialize a pixel-by-splat gradient tensor.

The default projection writes a 32-byte hot raster packet used by bin/sort/
raster and an 80-byte cold VJP packet used only by the per-splat 3D update.
The default backward reduces each block's pixel lanes in workgroup memory and
atomically adds a 12-float screen/conic/color/opacity gradient per splat. This
keeps the expensive camera/covariance/quaternion VJP out of the tile/splat
inner loop. The direct `tiled3d` selector preserves the previous monolithic
projection and repeated 3D VJP as a live correctness/performance control.

The default objective is:

```text
mean_pixel weight * (0.8 * abs(prediction - target) + 0.2 * (1 - SSIM))
```

The default SSIM uses the standard 11x11 Gaussian window with sigma 1.5,
reflected image boundaries, and `C1=0.01^2`, `C2=0.03^2`. Its analytic image
gradient passes finite differences, and the active Apple WebGPU forward,
objective, and selected parameter-family gradients pass the live tiled parity
harness. Nondefault SSIM radii remain normalized uniform-window probes and
change the objective.

The optional `Motion-Weighted Loss` ablation derives train weights from RGB
residual to each camera's temporal mean, caps them at 2x before normalization,
and normalizes them to mean one per frame. It changes neither the raster nor
the shared backward. Validation PSNR, SSIM, L1, and MSE are always ordinary
unweighted full-image metrics.

This ablation is off by default. At step 16,384, the unweighted run reached
`15.3/14.6 dB` train/heldout; 2x motion weighting reached `14.9/13.9 dB`, and
an earlier 4x run reached `15.3/14.3 dB`. It did not fix the plateau and
reduced heldout quality.

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
bounded to 6:1 in the tiled trainer. Initialization starts inside 3:1, and the
all-splat sampled fallback uses 4:1.

These are optimizer trust regions, not a penalty toward spheres. With scale as
standard deviation, 6:1 still permits a 36:1 covariance eigenvalue ratio and
fully trainable rotation. The cap prevents a noisy scale update from creating a
needle that spans many tiles, explodes pair work, becomes poorly conditioned,
and lets one primitive cover an edge instead of asking topology to represent
it. The cost is less freedom for legitimately thin structure.

A matched 12:1 ablation at step 16,384 produced `16.3/15.4 dB`
train/heldout and `0.518/0.254` SSIM, versus `16.2/15.5 dB` and
`0.514/0.261` at 6:1. The longer ellipses slightly improved train fit, slightly
hurt heldout quality, and touched more tiles, so 6:1 remains the default.

The `Motion Model` selector is only a trajectory-basis ablation:

- `Harmonic trajectory splats` adds one sinusoidal center component.
- `Linear trajectory splats` omits it.

Neither mode uses a native four-dimensional covariance. The precise name of
both modes is **trajectory-gated dynamic 3DGS**.

## Initialization And Topology

The exporter transforms the external SfM point cloud into the declared anchor
camera, keeps points visible from at least one training camera, and takes a
deterministic farthest-point subset. Initial scale and rotation come from local
point-cloud neighborhoods; opacity starts at 0.1.

The trainer uses fixed GPU capacity. The SPA initializes all 4,096 checked-in
SfM seeds and reserves 8,192 slots by default, then fills the second half
through splitting. This preserves the complete point scaffold while testing
the 1.19-splats-per-training-pixel capacity requested for the 96x72 raster.
Larger 16K, 24K, and 30K reserves are explicit scaling experiments; dormant
slots skip projection but still consume parameter, optimizer, clear, update,
readback, and preview-sort capacity. Lower slider values remain useful growth
ablations:

- when requested splats are below capacity, hidden slots are filled by
  split/recycle events beginning at step 600;
- after the reserved slots are filled, topology maintenance stops;
- selecting 4,096 initial splats uses the complete seed bank and grows to 8,192
  by step 26,100; no proxy-driven replacements occur after reserved capacity is
  full.

Growth is dynamic topology within a fixed allocation. It supports
relocation, spatial scale shrinkage, opacity-mass preservation, temporal
separation, and moment reset only while filling initially hidden slots. It is
not a dynamic buffer resize, residual/depth-guided densification, canonical
3DGS densification, or native 4DGS spatial-temporal pruning schedule. Proxy
recycling after fill was removed because it would replace 3,744 of 4,096 slots
by step 120,000 in the default run, erasing the SfM scaffold and producing
repeated split chains.

The global model and tile-local sort capacities are deliberately separate.
The internal model/preview path accepts 32,768 IDs, while each 8x8 tile sorts
at most 4,096 contributors. This keeps workgroup storage portable instead of
requiring more than a 32 KiB tile-local array. The SPA stops at a labeled 30K
stress preset because the full-cycle 32K Coffee Martini test overflowed.
`Tile Overflow` reports current and cumulative dropped tile/splat references;
both must remain zero for a run to be valid. The sort key uses 15 splat-ID
bits, so IDs 0 through 32,767 survive depth sorting.

Adam uses `beta1=0.9`, `beta2=0.999`, and `epsilon=1e-8`. The default-on,
toggleable schedule decays geometry learning rates by 100x and appearance
learning rates by 10x over 120,000 steps. Density statistics use 0.999 decay so
their memory spans the 272 camera/time-pair cycle.

### Background And Static Warmup

The SPA now defaults `Random Train BG` on for the tiled full-frame backend.
Each optimizer step hashes its step number once on the CPU, packs one
deterministic RGB underlay into the existing 160-byte uniform, and trains the
source-over result

```text
prediction = accumulated_splat_rgb + residual_transmittance * train_background
```

The background is shared by the complete image for that step. It is not hashed
per pixel, stored as an image, or sampled from a camera. The replay backward
reconstructs the suffix from final rendered RGB, so opacity and geometry see
the same underlay as the forward pass. The live WebGPU parity gate exercises
this path rather than merely checking shader source.

Validation, snapshot metrics, and all three live result panels still composite
over exact black. They therefore continue to expose true splat coverage instead
of hiding holes behind the train underlay. No camera-specific 2D background is
pasted behind the splats: doing so for `cam06` would leak heldout pixels and
make novel-view metrics meaningless. This follows the purpose of the optional
random-background training path in the
[original 3DGS implementation](https://github.com/graphdeco-inria/gaussian-splatting/blob/main/train.py)
while retaining black evaluation.

Decoded target RGB is clamped to `[0, 1]` and every target pixel is opaque.
Learned splat RGB is now clamped to the same range. The earlier `[0, 1.4]`
bound let a translucent overbright splat reproduce a bright target against
black without learning the corresponding opacity. Train and heldout alpha
coverage remain visible in the UI. There is no hidden coverage penalty.

This changed after the fixed-black run reached step 50,544 with only 59.9%
mean train alpha and 57.7% heldout alpha. With roughly 40% residual
transmittance, black was a material part of every prediction, not a harmless
rendering convention. Random train backgrounds break that opacity/color
shortcut, but they also change the training objective and can favor large
opaque floaters. The checkbox remains an explicit ablation, and black-eval
PSNR/SSIM/coverage are the comparison metrics. A separate coverage penalty was
not added in the same change; it would confound the first matched test.

Random backgrounds do not solve the other evidence from that run: all 4,096
slots were active, topology operations were zero, and 16% of splats were at the
6:1 aspect trust region. Residual-guided fixed-budget relocation remains the
next topology experiment if a fresh random-background run still plateaus.

The optional `Static Scene Warmup` uses those means for the first 2,048
optimizer steps, but only for the 17 training cameras. During that phase the
temporal gate is one and motion parameters are frozen; position, covariance,
rotation, RGB, and opacity build a static 3D scaffold. Training then restarts the
complete camera/time schedule on the real frames. Validation always renders the
actual train and heldout frames, never the temporal means.

It is off by default. A matched Coffee Martini smoke at step 16,384 favored no
warmup: `15.3/14.6 dB` train/heldout versus `15.1/14.2 dB` with warmup. The
mean-image phase also bakes the moving actor's temporal ghost into the scaffold,
which is the wrong default for this scene.

The live renderer now uses display-resolution footprint filtering, the same
`1/255` alpha support threshold and `0.99` alpha cap as training, and an exact
black clear. The old preview used the 72-pixel training height for filtering
even on a much taller canvas, which made small anisotropic splats look like
soft circles and exposed faint recycled copies that training had already
culled.

### Raster Filtering And Antialiasing

The current rasterizer projects the world covariance with the pinhole Jacobian
and adds an isotropic screen-space floor:

```text
C_screen = J R C_world R^T J^T + (0.3 / image_height)^2 I
```

Screen coordinates are normalized by image height, so this is a `0.3 px`
standard deviation, or `0.09 px^2` diagonal variance. The same constant is
shared by CPU parity, sampled training, tiled training, and live rendering.
The raster then evaluates that Gaussian at each pixel center, truncates at
three standard deviations (`q <= 9`), drops contributions below `1/255`, and
composites front to back.

This is a conservative EWA-style footprint floor, not complete alias-free
rendering. [EWA Splatting](https://www.cs.umd.edu/~zwicker/publications/EWASplatting-TVCG02.pdf)
provides the projection/filtering foundation. The
[original 3DGS rasterizer](https://github.com/graphdeco-inria/diff-gaussian-rasterization/blob/main/cuda_rasterizer/forward.cu)
uses the stronger `+0.3 px^2` dilation. [Mip-Splatting](https://openaccess.thecvf.com/content/CVPR2024/papers/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.pdf)
shows why bare 2D dilation changes apparent opacity and size: it adds a
determinant-compensated 2D Mip filter and a view-derived 3D smoothing
constraint. [Analytic-Splatting](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02597.pdf)
instead approximates the Gaussian integral over the pixel area.

The next honest antialiasing ablation is current filtering versus a complete
Mip-style 2D filter, including opacity compensation and its covariance
derivative in the shared backward, tested across train/display resolutions.
A forward-only compensation patch would make training and gradients disagree.
The 3D Nyquist constraint is a separate experiment because it needs each
primitive's maximum observed sampling frequency. Neither change should be
presented as a same-resolution convergence fix without a matched run.

### Why Not Softmax Splatting?

[Softmax Splatting](https://openaccess.thecvf.com/content_CVPR_2020/papers/Niklaus_Softmax_Splatting_for_Video_Frame_Interpolation_CVPR_2020_paper.pdf)
is a differentiable forward-warp operator for 2D video frame interpolation.
Its softmax resolves several source pixels landing on one target pixel. It is
not the depth-ordered transmittance model used by 3D Gaussian Splatting.
Replacing source-over alpha compositing with a softmax across Gaussians would
normalize away visibility and change both the rendering model and the shared
backward. It also does not integrate a footprint over a pixel, so it is not an
antialiasing method for this rasterizer.

It could be useful later in an optical-flow auxiliary loss or initialization
experiment, but it is not a convergence patch for this 3D renderer. The
higher-value standard additions are residual/view-space-gradient-guided
densification and pruning, view-dependent appearance, a verified train-only
SfM cloud, and a richer temporal parameterization.

## Runtime And Metrics

Training runs in a dedicated worker. It submits bounded eight-step bursts and
keeps no more than 32 GPU steps queued. Queue completion probes publish actual
completed throughput rather than command-submission speed.

Live rendering is capped at 20 GPU frames per second and shows two training
cameras plus the heldout camera at a looping time. It can be disabled without
stopping optimization.

The optimizer writes objective/L1/DSSIM into a 272-entry GPU ring every step.
Asynchronous readback runs every 256 requested steps and reports the latest raw
camera/time pair plus the mean of the most recent full 17-camera x 16-time
cycle. This avoids the old 256-versus-272 cadence alias, which sampled only 17
recurring phases and created a false 4,352-step ripple. It adds no optimizer
wait: the worker continues submitting work while the copy/map completes.

Every 8,192 steps, full metrics request an asynchronous parameter snapshot and
send it to a separate validation worker. The full-image raster and metric pass
never runs in the training worker.

The validation worker also reports active versus raster-dead slots,
dynamic/persistent counts, static-mix quantiles, endpoint temporal support,
anisotropy saturation, and per-family update RMS. These diagnostics never block
the optimizer pump.

Longitudinal validation uses all pixels from `cam04` and `cam09` over all 16
times, plus all pixels from heldout `cam06` over all 16 times. A second
center-time sweep evaluates every one of the 17 train cameras and reports the
weakest, median, and strongest PSNR. The charts retain full decimated history
and show:

- optimizer objective;
- full-image train and heldout MSE/MAE/PSNR;
- full-image channelwise 11x11 Gaussian SSIM;
- center, motion, scale, rotation, color, and opacity RMS deltas between
  validation snapshots.

This is a deterministic browser diagnostic, not the Python paper evaluator:
it currently omits LPIPS, and only the center-time camera sweep covers all 17
train cameras. A measured 4,096-splat validation pass took about 4.5 seconds on
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

Through 4,096 global splats, each tile reserves enough IDs to cover every
splat. The 8,192-splat path deliberately retains this 4,096-contributor
tile-local bound and reports violations through `Tile Overflow`; valid measured
runs must remain at zero. Active tile/splat pairs are compacted, backward
dispatch uses two workgroup dimensions, and pair-owned gradients accumulate
into one FP32 record per splat with compare/exchange atomics. Checkpoint stride
expands only enough to keep each storage binding within the device limit. On
the Apple M4's 128 MiB binding limit, the 384x288/4,096-splat plan is:

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

## July 30 8K Splat Result

The tiled backend now supports 8,192 global splats. The live SPA starts from
the 4,096 checked-in SfM seeds, allocates the 8K model once, and activates the
hidden half through opacity-preserving splits from step 600 through step
26,100. The tile-local sorter remains bounded at 4,096 entries, and its packed
depth key now preserves 13-bit splat IDs.

A synchronized 96x72 Apple M4 benchmark used packed-FP16 checkpoints, an 11x11
Gaussian SSIM window, 32 warmup steps, 128 measured GPU-drained steps, and
three fresh devices per count:

| Active splats | Median steps/s | Relative throughput | Tile overflow |
| --- | ---: | ---: | ---: |
| 4,096 | 350.2 | 100.0% | 0 |
| 8,192 | 188.3 | 53.8% | 0 |

The large tile-pair and checkpoint buffers stay fixed because tile capacity
does not grow with global capacity. Per-splat state does grow; for example, the
FP32 gradient accumulator rises from 0.375 MiB to 0.75 MiB. Giving the sort key
one more ID bit also reduces its positive-depth field from 20 to 19 bits.

The 8K benchmark activates a repeated copy of the 4,096-seed bank to stress
IDs above 4,095 and worst-case occupancy. It validates execution, finite loss,
and zero overflow, not reconstruction quality. The canonical growth path avoids
those visible duplicates, but still needs a matched convergence run to show
whether the added capacity recovers high-frequency detail.

Full repeats and memory plans are in
`benchmark_results/2026-07-30_tiled_8k_apple_m4.json`.

## July 30 20K-30K Scaling Result

The tiled allocation no longer inherits the sampled backend's
`view x time x splat` depth-order cache. That avoids 32.64 MB at 30K. It also
omits the unused sampled-gradient slab. At the current 96x72 raster, raw GPU
buffer capacity is therefore not the 20K-30K limit.

A synchronized full-cycle Apple M4 stress used 8 warmup steps followed by 272
measured GPU-drained steps. All splats were active by repeating the 4,096 seed
bank; this is deliberately a systems/occupancy stress, not an initialization
recommendation.

| Active splats | Steps/s | GPU buffers | Max tile | Total overflow |
| ---: | ---: | ---: | ---: | ---: |
| 20,000 | 99.5 | 31.4 MiB | 2,680 | 0 |
| 30,000 | 63.9 | 38.9 MiB | 3,993 | 0 |
| 32,768 | 58.8 | 40.9 MiB | 4,096 | 9,125 |

The 32K row is rejected. Its final camera/time pair happened to report zero
current overflow, but the new cumulative counter caught dropped references
earlier in the cycle. The 30K row completed cleanly but is close to the 4,096
tile bound, so the live cumulative counter remains a validity requirement as
optimization changes footprints.

The dominant next speed patch identified here was implemented and measured on
July 31: projected-space gradient staging now runs the expensive 3D VJP once
per splat rather than once per tile/splat pair. Other measured risks remain the
three-panel global preview bitonic sorts and the serial topology selector.
Full pre-fork data is in
`benchmark_results/2026-07-30_tiled_20k_30k_apple_m4.json`.

## July 31 WebGPU Kernel Fork Result

The fast tiled backend was forked behind explicit controls, compared against
the direct reference, and promoted only after active Apple WebGPU parity.
The accepted configuration is:

| Component | Fast default | Live control |
| --- | --- | --- |
| Raster backward | staged projected gradient, one 3D VJP/splat | repeated direct 3D VJP |
| Backward work | checkpoint block | tile/splat pair |
| Tile edge | 8 | 16 |
| Projection storage | 32 B hot + 80 B cold/splat | 192 B monolithic/splat |
| SSIM | exact separable 11x11 | direct 11x11 2D |
| Checkpoints | pixel-major packed FP16 | selectable |

One-step parity passes with maximum RGB error `1.19e-7`, objective error
`5.45e-8`, all 9 active gradient families accepted, and zero tile overflow.
The exact separable SSIM uses

```text
H = H_y H_x
H^T = H_x^T H_y^T
```

for the Gaussian window and its adjoint. It changes neither the objective nor
the reflected-boundary convention; it replaces a quadratic 2D neighborhood
walk with four one-dimensional passes and 80 bytes of scratch per pixel.

Final timings use an order-controlled protocol: both variants remain alive,
both are warmed, four equal measurement chunks alternate execution order, GPU
timestamp profiles alternate order, and the queue is drained only at
measurement boundaries.

The saved rows below use the sum of active pass timestamps for `GPU step`.
New artifacts also report first-pass-begin to last-pass-end as `gpuSpanMs`.
Phase contracts are explicit: direct `backward` includes its repeated 3D VJP,
while staged `backward` emits projected gradients and staged `update` contains
the one-per-splat 3D VJP plus Adam. The inner backward ratio is therefore a
useful location-of-savings diagnostic, not a matched standalone-kernel claim.
End-to-end throughput and complete GPU span are the primary comparisons.

| Fork | Matched workload | Throughput | GPU step | Inner phase | Memory |
| --- | --- | ---: | ---: | ---: | ---: |
| staged vs direct backward | 8K, 96x72 | 1.84x | 1.80x | 2.52x reported backward | -1.00 MiB |
| staged vs direct backward | 30K, 96x72 | 1.70x | 1.42x | 2.79x reported backward | -3.66 MiB |
| compact vs monolithic projection | 30K, 96x72 | 1.05x | 1.07x | packet/VJP aggregate | -2.29 MiB |
| separable vs 2D SSIM | 8K, 96x72 | 1.10x | 1.20x | 2.50x stats, 5.70x adjoint | +0.53 MiB |
| separable vs 2D SSIM | 8K, 192x144 | 1.28x | 1.32x | 2.65x stats, 3.66x adjoint | +2.11 MiB |
| separable vs 2D SSIM | 30K, 96x72 | 1.05x | 1.07x | 2.57x stats, 3.79x adjoint | +0.53 MiB |

The compact packet's two reversed-start runs improved throughput by 1.03x and
1.06x, so its modest gain is less sensitive to process order than the earlier
one-shot benchmark. Historical one-shot numbers remain in the artifact but
are explicitly non-headline evidence.

Rejected or parked forks are also part of the result:

- a shared pair packet was slower than lane-local accumulation;
- checkpoint strides 8 and 32 lost to 16 on this workload;
- block-major checkpoints lost to pixel-major;
- the 32K full camera/time cycle overflowed the 4,096-entry tile bound;
- one-shot control-then-candidate timing changed materially when order flipped,
  which motivated the alternating protocol.

Raw order-controlled artifacts live under
`benchmark_results/2026-07-31_interleaved/`. The consolidated result and
historical context are in
`benchmark_results/2026-07-31_wgpu_kernel_forks_apple_m4.json`. The full
scientist reflection, derivations, backtracks, and next falsification lanes are
in
`../../agent_notes/loose_notes/2026-07-31_03-36-30_browser_wgpu_kernel_forks_scientist_reflection.md`.

### Contention-qualified v3 rerun

The stronger `v3` protocol reran the 8K backward comparison after adding host
contention capture, round-CV validity, execution-position diagnostics, and a
five-second postflight GPU cooldown:

| Initial order | Staged wall speedup | Staged GPU-span speedup | Maximum round CV | Maximum position bias |
| --- | ---: | ---: | ---: | ---: |
| control first | 1.731x | 1.795x | 0.60% | 0.39% |
| candidate first | 1.752x | 1.724x | 1.00% | 0.45% |

Both artifacts have `validity.promotable=true`, finite loss, zero overflow,
preflight Apple GPU utilization of `0–2%`, postflight utilization of `0–3%`,
and quiet CPU/load/process-pressure checks. They are:

- `benchmark_results/runs/2026-07-31/backward_8k_control_first_v3_apple_m4.json`
- `benchmark_results/runs/2026-07-31/backward_8k_candidate_first_v3_apple_m4.json`

The headless lab also stopped constructing sampled-ray motion/static banks.
That preprocessing sorted candidates from roughly 1.9 million train pixels but
is not read by this full-frame benchmark, which fixes `motionWeighting=false`.
Calibrated targets, camera/time scheduling, initialization, objective, and WGSL
steps are unchanged.

Several runs were intentionally not promoted: preflight Apple GPU utilization
reached `69–96%` while media analysis was active; two autoruns timed out before
training because an invalid HTML number-step lattice silently blocked form
submission; and a 256-splat timer smoke exceeded the `10%` CV limit. Those
failures informed the harness but remain `/tmp` diagnostics rather than durable
performance evidence.

The same protocol now covers the important scaling lanes:

| Pair | Candidate wall speedup | Candidate GPU-span speedup | Pair wall drift | Max absolute throughput drift |
| --- | ---: | ---: | ---: | ---: |
| staged backward, 8K, 96x72 | 1.731–1.752x | 1.724–1.795x | 1.20% | 1.37% |
| staged backward, 30K, 96x72 | 1.818–1.818x | 1.741–1.751x | 0.03% | 0.28% |
| compact projection, 30K, 96x72 | 1.084–1.099x | 1.087–1.103x | 1.38% | 1.15% |
| separable SSIM, 8K, 192x144 | 1.283–1.288x | 1.280–1.292x | 0.36% | 0.34% |

All four pair summaries are promotable under maximum relative drifts of `5%`
for wall speedup, `10%` for GPU speedup, and `5%` for either variant's absolute
throughput. The 30K runs require a 4,096-entry tile capacity; a 1,024-entry
diagnostic dropped about 388K tile/splat pairs and was rejected before it could
be mistaken for a nearly 2x result. The 192x144 SSIM pair uses 512 measured
steps and nine profiles because the shorter 128-step pair sat too close to the
CV threshold.

Pair summaries and their source artifacts live together under
`benchmark_results/runs/2026-07-31/`.

## What The Numbers Do Not Prove

- The live SPA is slower than the isolated table because preview, worker
  scheduling, status publication, metric readback, and validation coexist with
  training. The final 4,096-active/8,192-capacity smoke reported 170.5
  completed steps/s at step 1,248; a prior 4,096-capacity observation was about
  250-280 steps/s.
- Historical 7.3, 584-824, and 793 steps/s observations belong to different
  sampled-ray kernels, splat counts, objectives, and UI schedules.
- No saved Metal measurement matches this complete browser step. Projection,
  raster-only, or synthetic Metal microbenchmarks cannot establish a
  WebGPU-versus-Metal percentage.
- Historical million-scale counts were frame/batch-expanded projected
  instances or pre-compaction segment records, not one million distinct
  trainable scene splats through a complete optimizer step.
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
2. residual/depth-guided densification and pruning, compared against the new
   fixed-topology default rather than the removed proxy recycler;
3. matched initialization, normalized-scale-bound, LR-family, and splat-capacity
   ablations;
4. full-image heldout PSNR, SSIM, LPIPS, and L1 on more than one scene and seed;
5. a complete calibrated dynamic-3DGS baseline before promoting native 4DGS or
   World Tubes to a selectable browser backend;
6. canonical byte targets plus shared host storage before presenting 384x288 as
   a normal SPA dataset mode rather than an isolated GPU benchmark.

The latest corrected diagnostic reached `16.2/15.5 dB` train/heldout and
`0.514/0.261` SSIM at step 16,384, versus `15.3/14.6 dB` and
`0.494/0.214` before fixed topology and bounded RGB. At step 32,768 it reached
`16.6/15.5 dB` and `0.537/0.257`. These are single-scene smokes, not rows in
the canonical paper standings.

See `research_notes/browser_4dgs_baseline.md` for the external native-4DGS
comparison contract.

See `research_notes/browser_trajectory_3dgs_plateau_audit_2026-07-29.md` for
the measured plateau diagnosis and prioritized paper-space comparison.
