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

### Hosted build

The current static build is published at:

```text
https://nbardy.github.io/dynaworld/
```

GitHub Pages cannot configure COOP/COEP response headers directly. The hosted
page therefore registers a same-origin isolation service worker and reloads
once on its first visit. Subsequent loads use SharedArrayBuffer for the target
bank and report `atomic SAB` in the runtime diagnostics. The `gh-pages` branch
contains only this directory's static subtree; the checked-in deployment
workflow can replace the legacy branch build when repository Actions are
available.

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

The checked-in presets share one calibration/split contract:

| Field | 96x72 default | 384x288 native 4x-linear |
| --- | --- | --- |
| Bundle | `coffee_martini_train17_holdout1.json` | `coffee_martini_train17_holdout1_384.json` |
| Scene | Neural 3D Video Coffee Martini | same |
| Raster | 96x72 | 384x288 |
| Times | All 300 synchronized frames at 30 fps (10.0 s), 16 resident at once | same full-rate timeline |
| Training cameras | 17 | 17 |
| Heldout cameras | `cam06` only | `cam06` only |
| Checked-in initialization | 4,096 train-visible external Ex4DGS XYZRGB points | same points |
| Anchor coordinates | `cam04` OpenCV camera frame | same frame |
| Pose convention | LLFF raw `[down, right, back]` to OpenCV `[right, down, forward]` (`v2`) | same convention |

Coffee Martini really is a short capture: the source videos contain exactly
300 frames at 30 fps, or 10.0 seconds. The old browser run used 16 samples
spread across those 10 seconds; it did not train at full frame rate. Both
presets now expose every source frame through 19 bounded temporal pages. A
normal page contains 16 frames and the final page contains 12.

Pages are interleaved across the complete clip rather than contiguous
half-second chunks. Therefore every resident 17-camera training cycle sees
the whole motion interval, while page rotation eventually visits every exact
source timestamp. Model time is normalized over observed frame centers, so
source frames 0 and 299 map to `t=0` and `t=1`.

The browser does not run COLMAP. The bundle exporter is a thin adapter over
`src/train/multicam_video_data.py`; it preserves the canonical manifest,
camera calibration, train/heldout split, synchronized frame indices, and
initialization provenance. Heldout pixels are never used by the optimizer.

The pose-source identifier is
`neural_3d_llff_opencv_relative_pinhole_v2`. Raw `poses_bounds.npy` rows store
camera basis columns as `[down, right, backwards]`; the loader reorders them to
OpenCV `[right, down, forwards]` before forming anchor-relative poses. The
superseded sign-only conversion is intentionally a different artifact identity.
The SPA rejects that legacy identity, non-finite or non-rigid camera matrices,
and a non-identity anchor transform before creating the trainer.

JSON numbers become `Float32Array` camera, seed, and optimizer buffers. Target
video is stored as 18 checked-in 384x288 H.264 streams and decoded into bounded
RGBA8 page banks; the original 16-frame RGBA8 atlases remain the no-video
fallback. Only the selected frame is decoded to `f32` on the GPU for raster
loss. Browser
normalization multiplies seed XYZ and every camera translation by the same
inverse-median-depth scale, so pixel projection is invariant. Those are native
LLFF scene units, not documented meters. A running worker owns its loaded
camera copy. Changing the resolution or pressing Reset reloads the SPA and
constructs a fresh dataset and trainer, which is also required after replacing
a bundle or pose convention on disk.

The checked-in bundle predates the provenance report contract. Its external
Ex4DGS cloud is filtered after loading to points visible from at least one
training camera, but the repository cannot prove which source images originally
created that cloud. In particular, it cannot prove that `cam06` was excluded.
The SPA therefore labels this initialization `unverified`; "train-visible" is
not the same claim as "constructed from train cameras only."

### Full-rate temporal paging

The 18 full-rate streams occupy 17,936,497 bytes (17.106 MiB) on disk. Eagerly
decoding the complete 384x288 corpus would require 2,388,787,200 bytes (2.225
GiB) of RGBA8 host memory before backgrounds or browser overhead. The paging
plan instead keeps at most the current and prefetched 16-frame pages:

| Raster | Two RGBA8 pages | FP32 camera means | Bounded total |
| --- | ---: | ---: | ---: |
| 96x72 | 15.188 MiB | 1.898 MiB | 17.086 MiB |
| 384x288 | 243.000 MiB | 30.375 MiB | 273.375 MiB |

Training still binds exactly one selected target image on the GPU. The next
page is decoded asynchronously while the optimizer runs in its dedicated
worker. A completed page is published through the existing shared dataset
boundary, the training worker swaps its resident frame bank in queue order,
and the validation worker receives the same page identity. The optimizer never
awaits video decode, `queue.onSubmittedWorkDone`, validation, or UI readback.

Decode currently uses asynchronous `HTMLVideoElement` plus canvas work on the
UI thread. That keeps GPU training nonblocking, but decode can still compete
for CPU time or cause a UI hitch on a loaded machine. A WebCodecs decode worker
is a future optimization only if timeline/long-task profiling shows this is a
material bottleneck; adding another thread without that evidence would make
the prototype harder to reason about.

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

The exporter transforms the external, provenance-unverified Ex4DGS point cloud
into the declared anchor camera, keeps points visible from at least one training
camera, and takes a deterministic farthest-point subset. Initial scale and
rotation come from local point-cloud neighborhoods; opacity starts at 0.1.

The trainer uses fixed GPU capacity. The SPA initializes all 4,096 checked-in
SfM seeds and reserves 8,192 slots by default, then fills the second half
through splitting. This preserves the complete point scaffold while testing
the 1.19-splats-per-training-pixel capacity requested for the 96x72 raster.
Larger 16K, 24K, and 30K reserves are explicit scaling experiments. Dormant
slots consume allocated parameter and optimizer memory, but active-prefix
dispatch now excludes them from training clear/update, preview draw/sort, and
validation telemetry. Lower slider values remain useful growth ablations:

- when requested splats are below capacity, hidden slots are filled by split
  events beginning at step 600;
- after the reserved slots are filled, topology maintenance stops;
- selecting 4,096 initial splats uses the complete seed bank and grows to 8,192
  by step 26,100; no proxy-driven replacements occur after reserved capacity is
  full.

The 30K preset is not a quality default. At 16 children per 100 steps it does
not fill until roughly step 162,400, after the default geometry schedule has
already decayed by about 100x. It is retained to measure scaling and long-run
topology behavior; use 8K for the calibrated convergence baseline.

Growth is dynamic topology within a fixed allocation. It supports parent/child
spatial separation, scale shrinkage, opacity-mass preservation, temporal
separation, and moment reset only while filling initially hidden slots. It does
not yet relocate a low-value live splat, resize a GPU buffer, use residual or
coarse-depth-guided births, reproduce canonical 3DGS pruning, or implement a
native 4DGS spatial-temporal pruning schedule. Proxy
recycling after fill was removed because it would replace 3,744 of 4,096 slots
by step 120,000 in the default run, erasing the SfM scaffold and producing
repeated split chains.

The split score already avoids same-view gradient cancellation: each covered
pixel contributes the magnitude of its projected-mean gradient before the tile
reduction. This is an [AbsGS](https://arxiv.org/abs/2404.10484)-style
non-cancelling statistic and inherently gives
larger pixel footprints more evidence, but it is not a verbatim implementation
of AbsGS's separate absolute x/y sums and threshold calibration.

`Near-Camera Floater Guard` is default on and remains a reset-time A/B control.
Following [Pixel-GS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf),
it multiplies only that density statistic by

```text
clip((camera_depth / (0.37 * camera_scene_radius))^2, 0, 1)
```

where camera-scene radius is `1.1 * max distance from a training-camera center
to their mean`. It does not scale the renderer, image loss, or Adam gradient.
This suppresses extra births close to a camera without making the optimization
VJP disagree with the forward pass. It is paper-backed but newly integrated;
quality claims still require the matched on/off run described below.

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
their memory spans a normal 272-pair resident cycle (17 cameras x 16 frames);
the final 12-frame page uses 204 pairs.

### Background And Static Warmup

The SPA now defaults `Random Train BG` on for the tiled full-frame backend.
Each optimizer step hashes its step number once on the CPU, packs one
deterministic RGB underlay into the existing 192-byte uniform, and trains the
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

Live rendering is capped at 15 GPU frames per second and normally shows two
training cameras plus the heldout camera at a looping time. It can be disabled
without stopping optimization.

The primary page order is comparison, key quality, then configuration. The
calibrated 3x2 camera matrix and metric bands run edge to edge without outer
card gutters or rounded framing. Compact viewports retain the four
train/heldout quality readouts and Loss, PSNR, and SSIM histories before the
reset-sensitive controls and full operational diagnostic grid.

### Interactive result camera

The first screen is a three-column comparison matrix: calibrated ground truth
on the first row and the matching WebGPU result on the second. Every GT and
result cell controls the render-only camera in its column. The GT image stays
fixed as the calibrated reference while the result label gains `orbit` after
the camera moves. Left drag orbits, Shift-left/middle/right drag pans, the wheel
dollies, and double-click restores that column; the toolbar reset restores all
three calibrated cameras.

The interactive views use the same projection, depth sort, Gaussian raster,
temporal model, display filtering, and WGSL render pipeline as the calibrated
results. No external 3DGS viewer is loaded.

The orbit pivot starts on the selected camera's principal ray at the median
positive seed depth. Each interaction constructs a rigid OpenCV look-at camera
with +X right, +Y down, and +Z forward. The packer applies the trainer's global
geometry scale to translation only, preserving projection exactly.

Training and validation keep the immutable canonical camera buffer. Rendering
binds a separate copy with three additional preview slots, and only those slots
are rewritten during interaction. Thus dragging cannot change camera sampling,
losses, gradients, or heldout metrics. A moved result is useful for inspecting
geometry but is not a calibrated novel-view metric; double-click before making
pixel-aligned comparisons with its GT cell.

### Progressive and native 4x-linear resolution modes

`Training Resolution -> Progressive 96 -> 384` is the default. It trains the
coarse native bundle through step 8,192, preloads the native 384x288 bundle in
parallel, and then performs one bounded worker-owned transition. The worker
drains already-submitted GPU work, snapshots parameters, both Adam moments,
density statistics, active topology, cumulative tile diagnostics, the original
initialization, and the global step, rebuilds only resolution-dependent trainer
resources, restores that state, reattaches validation, and resumes if the run
was active. This is an intentional one-time pause rather than a periodic train
loop synchronization point.

The transition rejects changes in camera calibration, train/heldout split,
frame indices, seed geometry, primitive schema, capacity, or topology schedule
before it writes restored GPU state. Loss, PSNR, and SSIM retain their complete
history; the dashed cyan/blue vertical marker identifies the one resolution
handoff. It is not an evaluation stall, topology event, optimizer reset, or
page switch.
The manual `96 x 72` and `384 x 288` choices still initialize directly at one
resolution and therefore remain useful controls.

`Training Resolution -> 384 x 288` reloads the SPA with native-resolution
pages decoded from the 18 full-rate streams. It is four times wider and taller
than the default, so every train step evaluates 110,592 pixels instead of
6,912. This is not an upsample of the 96x72 browser bank.

At 4,096 initial splats, 8,192 capacity, packed-FP16 checkpoints, and the fast
tiled backend, the live app reports 97.6 MiB of GPU buffers. The steady host
timeline is bounded to 243.0 MiB for current/prefetched RGBA8 pages plus 30.375
MiB of FP32 camera means. The complete eager target corpus would be 2.225 GiB
and is never allocated. The largest GPU binding remains the 54 MiB
transmittance checkpoint buffer, below the Apple adapter's 128 MiB binding
limit.

A matched 2026-07-31 Apple-browser smoke, with a 15 Hz free-camera preview and
no tile overflow, measured about 309 completed steps/s at 96x72 and 130
completed steps/s at 384x288. Sixteen times the pixels cost about 2.4x
throughput in that live configuration because splat projection, sorting,
optimizer work, and scheduling do not scale with image area. Resolution is
therefore relatively cheap, but not free; occupancy and splat count can change
that ratio. The sampled-ray control remains disabled at 384x288 because that
legacy backend still binds the complete target tensor.

A 2026-08-03 live full-rate progressive smoke selected source frame 202 and
later frame 72, crossed to 384x288 at step 10,688 after asynchronous preload,
preserved trained parameters, Adam state, topology, and global step, and
continued past step 12,480 without a browser warning or tile overflow. The host
preflight failed promotion because CPU load, competing processes, Apple-GPU
load, and swap occupancy were all above the benchmark limits. The live rate is
therefore only a continuity diagnostic, not a replacement throughput baseline.

The optimizer writes objective/L1/DSSIM into a resident-cycle GPU ring every
step. A normal page has 272 entries and the final page has 204.
Asynchronous readback runs every 256 requested steps and reports the latest raw
camera/time pair plus the mean of the most recent complete resident cycle. This
avoids the old 256-versus-272 cadence alias, which sampled only 17 recurring
phases and created a false 4,352-step ripple. It adds no optimizer wait: the
worker continues submitting work while the copy/map completes.

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

### Bounded camera-stress diagnostic

The validation worker also evaluates a deterministic camera-stress stencil at
the center time on two representative train cameras and the heldout camera.
It is deliberately capped at 48 pixels high and runs beside, never inside, the
continuous WebGPU optimizer queue.

Two kinds of perturbation must not be conflated:

- **Optical zoom/shift:** focal length changes by `1.05x` or `1/1.05x`, and
  the principal point shifts by `+/-1.5%` on each axis. These preserve the
  captured camera center and orientation, so the real frame can be exactly
  crop-resampled into a valid target. `Zoom/Shift PSNR` is therefore a real
  pixel metric, reported as train / heldout.
- **Physical pose:** dolly by `+/-3%` of pivot distance, translate laterally by
  `+/-1.5%` of camera-rig radius, or orbit by `+/-2 degrees`. No captured target
  exists at those poses. The UI consequently reports geometry-risk indicators,
  not invented PSNR: near-camera alpha contribution, alpha from splats whose
  opacity-aware support rectangle covers at least 25% of the image, and
  coverage-weighted contributor depth spread `sigma/z`.

The risk indicators are useful for detecting the translucent-cloud failure
seen under small camera motion, but they are not calibrated geometry error and
have no promotion threshold yet. They use only the learned splats, known camera
calibration, and captured RGB images; no monocular or foundation-model depth
prior is involved.

The diagnostic changes no loss, gradient, topology, or default regularizer.
Its purpose is to make the failure measurable before choosing an intervention.
The first candidate intervention is multiview, transmittance-aware contribution
tracking followed by fixed-budget relocation of persistently unsupported
splats; that remains an ablation, not a claimed default.

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
still rejects 384x288. The tiled SPA now keeps canonical targets in shared
RGBA8 storage and decodes only the active target page to FP32 at the GPU
boundary. Camera temporal backgrounds remain FP32 because they are computed
values rather than source PNG samples.

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

### 2026-08-03 temporal VJP correction

The fast staged backward had one material correctness bug. Forward opacity
uses

```text
dynamic_gate = temporal_floor + (1 - temporal_floor) * temporal_kernel
time_weight = mix(dynamic_gate, 1, static_mix)
```

but the staged VJP reconstructed `time_weight` as `dynamic_gate`. With the
default static-heavy initialization (`static_mix=0.92`), opacity and temporal
gradients could be much smaller than the derivative of the image actually
rendered. The backward now reconstructs the identical mixed gate. Scale LR
also follows geometry's 100x decay while retaining its former step-zero value,
so late training cannot keep exchanging detail for larger blurred footprints.

The live Apple WebGPU parity gate now includes the production static-mix
fixture and passes with maximum RGB error `1.1920929e-7`, objective absolute
error `2.2585871e-7`, all 9 intended gradient families active, and zero tile
overflow. These are numerical-correctness results, not quality metrics.

### Novel-view floaters: diagnosis and paper-backed next step

The corrected LLFF/OpenCV camera contract reduced median epipolar error from
about 60 pixels to 0.48 pixels, and a fresh calibrated `cam06` run reached
27.0 dB / 0.909 SSIM. That makes a current gross camera-axis or world-scale bug
unlikely. The free orbit is a separate, unscored extrapolation surface and can
travel far outside the calibrated camera hull, so orbit artifacts must not be
reported as heldout-camera regression without a matched camera.

The strongest remaining code-level cause is topology allocation. Current
splits use a non-cancelling per-pixel screen-gradient statistic, alpha, and
velocity gradient, place a child beside its parent, and stop once capacity is
full. Pixel-GS depth scaling now suppresses near-camera split evidence. The
policy still does not birth from explicit image residual plus coarse depth,
measure cross-view support, or relocate low-value live splats after capacity is
full. The current trajectory 3DGS also keeps covariance, color, and opacity
constant over time and has no local-motion rigidity term.

The relevant paper interventions are deliberately ranked rather than mixed:

1. [AbsGS](https://arxiv.org/abs/2404.10484) motivates the active
   non-cancelling pixel-gradient score. The browser's scalar sum-of-magnitudes
   is close but not identical to separate absolute x/y accumulation, so a
   matched exact-statistic ablation remains valid.
2. [Pixel-GS](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/02926.pdf)
   motivates the newly integrated `gamma_depth=0.37` near-camera density
   scaling. It is default on but not yet promoted by a matched quality run.
3. [SpacetimeGS](https://openaccess.thecvf.com/content/CVPR2024/html/Li_Spacetime_Gaussian_Feature_Splatting_for_Real-Time_Dynamic_View_Synthesis_CVPR_2024_paper.html)
   uses training error plus coarse depth to guide births, while
   [3DGS-MCMC](https://arxiv.org/abs/2404.09591) supports fixed-budget
   relocation. Together they are the closest match to the remaining allocation
   failure without unbounded spawning.
4. [Mip-Splatting](https://openaccess.thecvf.com/content/CVPR2024/html/Yu_Mip-Splatting_Alias-free_3D_Gaussian_Splatting_CVPR_2024_paper.html)
   supplies the principled zoom/resolution experiment: 3D smoothing plus a
   determinant-compensated 2D Mip filter. The current 0.3-pixel sigma floor is
   display filtering, not complete Mip-Splatting.
5. [Dynamic 3D Gaussians](https://arxiv.org/abs/2308.09713) supplies a local
   rigidity prior, but it should be enabled only after diagnostics show that a
   floater moves incorrectly over time rather than merely appearing from a new
   camera.
6. [StopThePop](https://doi.org/10.1145/3658187) addresses view-dependent
   sorting pops, not stationary unsupported geometry. It is appropriate only
   if a fixed-model orbit trace shows camera-motion popping.
7. Sparse-view methods such as
   [DropoutGS](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DropoutGS_Dropping_Out_Gaussians_for_Better_Sparse-view_Rendering_CVPR_2025_paper.html),
   [CoMapGS](https://openaccess.thecvf.com/content/CVPR2025/html/Jang_CoMapGS_Covisibility_Map-based_Gaussian_Splatting_for_Sparse_Novel_View_Synthesis_CVPR_2025_paper.html),
   and [DepthSplat](https://openaccess.thecvf.com/content/CVPR2025/html/Xu_DepthSplat_Connecting_Gaussian_Splatting_and_Depth_CVPR_2025_paper.html)
   are useful evidence for uncertainty, covisibility, and depth priors, but
   Gaussian dropout is not a justified default for this 17-camera dynamic run.

The next implementation should record deterministic orbit alpha/depth traces
and residual-weighted per-splat contribution across the camera cycle. It can
then relocate low-contribution capacity toward high-error pixels with
multi-view support. That experiment follows the now-toggleable Pixel-GS guard;
it should not mix Mip filtering, dropout, and rigidity into the same run.

The highest-value remaining evidence is:

1. run the corrected full-rate baseline with `Near-Camera Floater Guard` on
   versus off under an identical pair order and step budget;
2. compare the verified 768-point train-only cloud plus growth against the
   legacy unverified 4,096-point seed under matched settings, or produce a
   denser verified cloud with a stronger matcher;
3. residual/depth-guided relocation and pruning, compared against the new
   fixed-topology default rather than the removed proxy recycler;
4. complete Mip-Splatting filtering, measured on deterministic zoom paths as
   well as calibrated cameras;
5. matched initialization, normalized-scale-bound, LR-family, and splat-capacity
   ablations;
6. full-image heldout PSNR, SSIM, LPIPS, and L1 on more than one scene and seed;
7. a complete calibrated dynamic-3DGS baseline before promoting native 4DGS or
   World Tubes to a selectable browser backend;
8. rerun the quality and throughput matrix at both checked-in resolutions and
   multiple splat capacities; the 384x288 mode is now functional, but one live
   smoke is not a convergence baseline.

The latest corrected diagnostic reached `16.2/15.5 dB` train/heldout and
`0.514/0.261` SSIM at step 16,384, versus `15.3/14.6 dB` and
`0.494/0.214` before fixed topology and bounded RGB. At step 32,768 it reached
`16.6/15.5 dB` and `0.537/0.257`. These are single-scene smokes, not rows in
the canonical paper standings.

See `research_notes/browser_4dgs_baseline.md` for the external native-4DGS
comparison contract.

See `research_notes/browser_trajectory_3dgs_plateau_audit_2026-07-29.md` for
the measured plateau diagnosis and prioritized paper-space comparison.

See
`research_notes/browser_full_rate_paging_and_novel_view_roadmap_2026-08-03.md`
for the paging derivation, temporal VJP backtrack, paper intervention matrix,
and ordered A0-A5 floater ablation plan.
