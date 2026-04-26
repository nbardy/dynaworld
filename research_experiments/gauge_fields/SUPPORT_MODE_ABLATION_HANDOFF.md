# Gauge Field Support-Mode Ablation Handoff

Date: 2026-04-26

This note summarizes the implementation pass that split the gauge-field renderer
into swappable support representations, added a calibrated multicamera held-out
view lane, and added a direct free-dynamic 3DGS control.

The short version: we now have a reusable ablation surface for asking whether
the persistent transported material field is learning usable 3D, or just
painting the source camera. The early held-out-camera results are close enough
that none of the representations should be thrown out yet.

## What Changed

The main gauge trainer is still:

```text
research_experiments/gauge_fields/train.py
```

It now supports three side-by-side material support modes through
`model.support_mode`:

| mode | meaning | current role |
| --- | --- | --- |
| `screen_disk` | projected isotropic disk, the original point-sprite baseline | fastest control |
| `oriented_slab` | persistent thin 3D slab, projected through the camera Jacobian | surface-biased support |
| `rank_adaptive_metric` | persistent PSD 3D metric, transported by a local deformation Jacobian | universal point/curve/surface/volume-like support |

All three modes share the same high-level material field:

```text
persistent canonical element positions
persistent color / opacity / radius-like scale
shared low-rank time transport
RGB / alpha / depth / X-map outputs
```

The difference is only how a material element becomes a projected pixel kernel.
That makes the ablation cleaner than introducing separate training loops for
each representation.

## Renderer Behavior

The renderer path was generalized from circular screen kernels to projected
covariance kernels.

`screen_disk`:

```text
screen covariance = scalar pixel radius * identity
```

`oriented_slab`:

```text
canonical slab scales + slab rotation
world covariance = R diag(r1^2, r2^2, h^2) R^T
screen covariance = J_camera world_covariance J_camera^T
```

`rank_adaptive_metric`:

```text
canonical PSD metric from diagonal/off-diagonal parameters
local deformation Jacobian from fixed canonical KNN
world covariance = J_deform metric J_deform^T
screen covariance = J_camera world_covariance J_camera^T
```

The implementation includes covariance hardening:

```text
bounded log-scales
nan-to-num guards
screen eigenvalue clamp
zero-axis-angle Rodrigues fix
```

The renderer also supports:

```text
render.opacity_transfer = "linear" | "optical_thickness"
```

The current benchmark configs use the simple linear path unless otherwise noted.

## Multicamera Held-Out Lane

The gauge trainer can now use:

```text
data.frame_source = "multicam_val"
```

The first calibrated held-out sample is:

```text
deepview_03_Dog_camera_0001_to_camera_0015
```

This trains on `camera_0001` and evaluates novel-view synthesis on
`camera_0015`.

The loader parses DeepView `models.json`, converts the DeepView/OpenGL-ish
camera convention into the gauge renderer's source-relative +Z camera frame, and
stores held-out camera tensors in the checkpoint. Outputs include:

```text
heldout_preview.png
heldout_side_by_side.mp4
heldout_eval_psnr / heldout_eval_l1
heldout_xmap_* diagnostics for gauge models
```

Caveat: DeepView cameras are fisheye. This lane currently uses a pinhole
approximation, so the metrics are useful for ranking experiments but should not
be treated as final camera-model truth.

## Direct 3DGS Control

A direct free-dynamic splat control was added:

```text
research_experiments/gauge_fields/train_splat_baseline.py
```

Config:

```text
src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc
```

This baseline uses the same DeepView source/held-out pair, but it is not a
persistent material-gauge model:

```text
2048 splats per frame
separate per-frame splat banks
direct optimized xyz / color / opacity / scale / rotation
no encoder
no implicit camera
no X-map diagnostic
```

It is a useful control because it answers: "Can ordinary free dynamic splats do
as well on the same train/held-out view split?"

## Ablations Set Up

### Source-View Support-Mode Benchmark

Artifact:

```text
outputs/gauge_fields/support_mode_benchmark_250step/summary.md
```

This is the original single-camera/source-view comparison. It is useful for
checking whether the support modes can fit the training camera, but it is not a
novel-view test.

| mode | steps | elements | eval PSNR | eval L1 | X-map occ |
| --- | ---: | ---: | ---: | ---: | ---: |
| `oriented_slab` | 250 | 2048 | 20.6312 | 0.0554 | 0.2910 |
| `screen_disk` | 250 | 2048 | 20.3144 | 0.0582 | 0.2544 |
| `rank_adaptive_metric` | 250 | 2048 | 19.8992 | 0.0606 | 0.2078 |

Read this result narrowly: slab fit the source camera best here, but source-view
PSNR is not the decision metric.

### DeepView Held-Out-Camera Benchmark

Artifact:

```text
outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/summary.md
```

All rows below use:

```text
DeepView 03_Dog
train camera = camera_0001
held-out camera = camera_0015
16 frames
128 px render size
80 optimization steps
2048 rendered primitives per frame
initial radius / scale = 0.035
```

| representation | train PSNR | train L1 | held-out PSNR | held-out L1 | X-map occ | held-out X-map occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `free_dynamic_3dgs` | 20.5017 | 0.0661 | 9.7392 | 0.2357 | n/a | n/a |
| `screen_disk` | 24.6535 | 0.0381 | 9.6479 | 0.2402 | 0.1409 | 0.1394 |
| `rank_adaptive_metric` | 24.2903 | 0.0389 | 9.5662 | 0.2412 | 0.1682 | 0.1560 |
| `oriented_slab` | 25.0395 | 0.0354 | 9.3344 | 0.2478 | 0.2004 | 0.1995 |

Interpretation:

```text
source-view ranking: slab > screen > rank > free 3DGS
held-out ranking: free 3DGS > screen > rank > slab
```

This is exactly why the held-out-camera lane matters. The representation that
best fits the source camera did not win the first novel-view score. The held-out
PSNR differences are small, so this is not a final verdict.

## Primitive And Parameter Accounting

The 80-step DeepView comparison above used the same number of rendered
primitives per frame: 2048.

That is not the same as equal parameter count.

| representation | primitive semantics | active/effective params | registered trainable params |
| --- | --- | ---: | ---: |
| `screen_disk` | 2048 persistent elements shared across time | 114,944 | 139,520 |
| `oriented_slab` | 2048 persistent elements shared across time | 125,184 | 139,520 |
| `rank_adaptive_metric` | 2048 persistent elements shared across time | 125,184 | 139,520 |
| `free_dynamic_3dgs` | 2048 splats per frame, separate over 16 frames | 458,752 | 458,752 |

The gauge model class currently registers the support tensors for all gauge
modes, so registered parameter count is equal across the three gauge rows. The
active/effective count is the cleaner accounting of parameters that actually
affect each representation's render path.

At 16 frames and 16 motion bases, rough active parameter matching to the
free-dynamic 3DGS row is:

| gauge mode | active-param matched element count |
| --- | ---: |
| `screen_disk` | about 8192 |
| `oriented_slab` | about 7516 |
| `rank_adaptive_metric` | about 7516 |

So yes, a fairer next comparison should include both:

```text
same primitive count
same active parameter budget
```

They answer different questions.

## Speed Notes

These are conservative 5-step end-to-end MPS timings. They include trainer
startup, optimization, final eval, and media/checkpoint writing, so do not treat
them as pure renderer step time.

| run | primitive budget | real time |
| --- | ---: | ---: |
| `screen_disk` | 2048 elements | 27.65 s |
| `screen_disk` | 8192 elements | 43.40 s |
| `oriented_slab` | 7516 elements | 172.20 s |
| `rank_adaptive_metric` | 7516 elements | 171.43 s |
| `free_dynamic_3dgs` | 2048 splats/frame | 51.16 s |

The useful read:

```text
screen_disk scales reasonably to the active-param-matched budget
slab/rank are much slower in the current pure Torch implementation
free_dynamic_3dgs is slower than 2048 screen_disk but faster than matched slab/rank
```

Do not overread the exact seconds. This should be followed with a cleaner timing
harness that reports train-step wall time and render-only wall time separately.

## Reusable Tools

### Gauge Trainer

```text
research_experiments/gauge_fields/train.py
```

Use for:

```text
screen disk / slab / rank-adaptive support comparisons
single-camera source-view overfit
DeepView held-out-camera evaluation
X-map and projection diagnostics
cheat-probe-compatible checkpoints
```

### Direct Splat Baseline

```text
research_experiments/gauge_fields/train_splat_baseline.py
```

Use for:

```text
free dynamic 3DGS control
same DeepView held-out-camera split
same summary tooling as gauge runs
```

### Sweep Config Generator

```text
research_experiments/gauge_fields/make_sweep_configs.py
```

Use for generating families of gauge configs over:

```text
support modes
element counts
radii
alpha logits
step budgets
```

It normalizes configs through the gauge config loader so generated files stay
close to the trainer schema.

### Run Summarizer

```text
research_experiments/gauge_fields/summarize_runs.py
```

Use for:

```text
aggregating final metrics.json files
sorting by train or held-out metrics
writing Markdown and JSON summaries
mixing gauge and splat baseline rows
```

It understands:

```text
support_mode
model.num_elements
model.num_splats
heldout_eval_*
X-map metrics when present
```

### Cheat Probe Tool

```text
research_experiments/gauge_fields/cheat_probe_material_gauge.py
```

Use for:

```text
radius perturbation
opacity perturbation
depth/transport-style probes
support-aware checkpoint loading
strict state dict loading
```

This remains gauge-only. The direct 3DGS baseline does not expose the same
material X-map contract.

## Key Configs

Multicamera DeepView gauge configs:

```text
src/train_configs/local_mac_gauge_fields_multicam_deepview_screen_disk_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_multicam_deepview_oriented_slab_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_128_16f_2048el.jsonc
```

Direct splat baseline config:

```text
src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc
```

Source-view support-mode configs:

```text
src/train_configs/local_mac_gauge_fields_material_surfel_motion_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_oriented_slab_motion_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_motion_128_16f_2048el.jsonc
```

Smoke configs:

```text
src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc
src/train_configs/local_mac_gauge_fields_oriented_slab_smoke_32_2f_32el.jsonc
src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc
```

## Rerun Commands

Gauge held-out-camera run:

```bash
.venv/bin/python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_screen_disk_128_16f_2048el.jsonc \
  --device mps \
  --steps 80 \
  --no-wandb \
  --output-dir outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/screen_disk
```

Swap the config and output directory for:

```text
oriented_slab
rank_adaptive_metric
```

Direct free-dynamic 3DGS run:

```bash
.venv/bin/python research_experiments/gauge_fields/train_splat_baseline.py \
  src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc \
  --device mps \
  --steps 80 \
  --no-wandb \
  --output-dir outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/free_dynamic_3dgs
```

Summarize a completed benchmark:

```bash
.venv/bin/python research_experiments/gauge_fields/summarize_runs.py \
  'outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/*' \
  --sort-by heldout_eval_psnr \
  --out-md outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/summary.md \
  --out-json outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/summary.json
```

Generate a support-mode sweep:

```bash
.venv/bin/python research_experiments/gauge_fields/make_sweep_configs.py \
  --base-config src/train_configs/local_mac_gauge_fields_multicam_deepview_screen_disk_128_16f_2048el.jsonc \
  --support-modes screen_disk oriented_slab rank_adaptive_metric \
  --elements 2048 4096 8192 \
  --radii 0.025 0.035 0.050 \
  --alpha-logits 0.0 \
  --steps 80
```

Run cheat probes on a gauge checkpoint:

```bash
.venv/bin/python research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  --checkpoint outputs/gauge_fields/multicam_deepview_support_mode_benchmark_80step/screen_disk/checkpoint.pt \
  --output-dir outputs/gauge_fields/cheat_probes/screen_disk \
  --device mps \
  --probe all
```

## What Not To Overclaim

Do not claim that the current implementation proves robust 3D or novel-view
synthesis. The first DeepView held-out view is a better gate than source-view
PSNR, but it is still one scene, one source camera, one target camera, one short
schedule, and a pinhole approximation of a fisheye camera.

Do not throw out slab or rank-adaptive support just because the first held-out
PSNR is lower. They have different tuning surfaces, different coverage behavior,
and much heavier current implementations. Small radius, opacity transfer,
regularization, and element-count changes could change the ranking.

Do not treat `free_dynamic_3dgs` as a matched semantic baseline. It has far more
active parameters at the same displayed primitive count because its splats are
free per frame.

## Best Next Runs

1. Same primitive budget:

```text
2048 elements/splats, 80 steps, same DeepView pair
```

Already done once. Repeat across more held-out target cameras and more scenes.

2. Same active parameter budget:

```text
screen_disk: 8192 elements
oriented_slab: 7516 elements
rank_adaptive_metric: 7516 elements
free_dynamic_3dgs: 2048 splats/frame
```

This answers the fairness concern around parameter count.

3. Cleaner timing benchmark:

```text
train-step time
render-only time
eval/media time separated
```

The current speed numbers are useful but too coarse.

4. Camera-model cleanup:

```text
replace the pinhole approximation with the actual DeepView fisheye model
```

This matters before making strong claims about held-out-camera quality.

5. Tune support-specific knobs:

```text
radius / coverage
opacity_transfer
slab thickness prior
metric spectrum regularization
Jacobian KNN count
```

The first fair conclusion is not "which representation wins"; it is that the
repo now has the machinery to run that comparison honestly.
