# Gauge-Field Support Modes

## Context

The chief-scientist note reframed the material-gauge experiment honestly: the
current model is a persistent transported point/disc field, not yet a true
surfel/surface/fluid primitive. The requested implementation was to compare the
current screen-disk baseline with two stronger transported support laws:
oriented slabs and rank-adaptive metrics.

## What changed

- Kept the current `screen_disk` path as the default baseline in
  `research_experiments/gauge_fields/train.py`.
- Added `model.support_mode` with three values:
  - `screen_disk`
  - `oriented_slab`
  - `rank_adaptive_metric`
- Added persistent support parameters to `MaterialSurfelField`:
  - `slab_log_scales` for thin slab support.
  - `slab_raw_rot` for per-element canonical slab orientation, initialized
    fronto-parallel by default.
  - `metric_log_diag` and `metric_offdiag` for PSD rank-adaptive support.
- Added fixed canonical KNN neighborhoods and an identity-plus-displacement
  local Jacobian estimate. The identity term matters because the initialized
  material points are often close to a plane, so the normal direction is
  underconstrained by raw neighbor least squares.
- Generalized the renderer from circular projected disks to projected 2D
  covariance kernels while preserving RGB/alpha/depth/X-map outputs and the
  same chunked `elements x pixels` loop.
- Added `render.opacity_transfer` with default `linear` so the old baseline is
  not silently changed. `optical_thickness` is available for later opacity-law
  tests.
- Updated cheat probes so radius/opacity probes also affect slab/metric support
  parameters and checkpoint loading preserves `support_mode`.
- Updated summarization to include `support_mode` and projected anisotropy.
- Added side-by-side smoke and 16-frame motion configs for `oriented_slab` and
  `rank_adaptive_metric`.

## Verification

Syntax:

```bash
.venv/bin/python -m py_compile \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  research_experiments/gauge_fields/make_sweep_configs.py \
  research_experiments/gauge_fields/summarize_runs.py
```

CPU smokes:

```bash
.venv/bin/python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_material_surfel_smoke_32_2f_32el.jsonc \
  --device cpu --no-wandb --output-dir /tmp/gauge_fields_screen_disk_smoke_test

.venv/bin/python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_oriented_slab_smoke_32_2f_32el.jsonc \
  --device cpu --no-wandb --output-dir /tmp/gauge_fields_oriented_slab_smoke_test

.venv/bin/python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc \
  --device cpu --no-wandb --output-dir /tmp/gauge_fields_rank_adaptive_metric_smoke_test
```

MPS smoke:

```bash
.venv/bin/python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc \
  --device mps --no-wandb --output-dir /tmp/gauge_fields_rank_adaptive_metric_smoke_mps_test
```

All smokes completed and wrote output directories. The one-step comparison table
showed `support_mode` and `projection_anisotropy_p95` correctly through
`summarize_runs.py`.

The first oriented-slab smoke exposed NaNs from the exact zero axis-angle
initialization. The fix was to clamp the Rodrigues helper away from the singular
`sqrt(0)` derivative while still using the small-angle polynomial branch. The
oriented-slab CPU and MPS smokes passed after that.

Cheat-probe loading was also smoked with:

```bash
.venv/bin/python research_experiments/gauge_fields/cheat_probe_material_gauge.py \
  --checkpoint /tmp/gauge_fields_rank_adaptive_metric_smoke_test2/checkpoint.pt \
  --output-dir /tmp/gauge_fields_rank_adaptive_metric_probe_smoke \
  --device cpu \
  --probe radius_inflate \
  --no-video
```

After the three-agent review, the implementation was tightened again:

- Support log-scales are bounded before exponentiation.
- Projected covariance matrices are sanitized and eigenvalue-clamped to the
  configured pixel-radius range to avoid `inf * 0` covariance collapse.
- Rank-adaptive radius probes now scale `metric_offdiag` as well as diagonal
  Cholesky entries.
- Slab radius probes scale only the two in-plane axes, leaving thickness alone.
- Checkpoint loading now allows missing support parameters only for legacy
  `screen_disk` checkpoints; non-screen support-mode mismatches fail loudly.
- Sweep generation now defaults to all three support modes, normalizes configs
  through `gauge_config`, includes the step budget in run slugs, and uses
  support-mode-specific W&B tags.
- Docs now avoid overclaiming: `oriented_slab` is thin-initialized but not hard
  surface-constrained, and `rank_adaptive_metric` is a full-rank metric
  candidate until rank/eigenvalue sparsity is added.

Extra verification:

```bash
# Instantiated slab/metric models, filled support logs with 100, rendered, and
# confirmed rgb/alpha/depth/xmap stayed finite.
.venv/bin/python - <<'PY'
...
PY

.venv/bin/python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_oriented_slab_smoke_32_2f_32el.jsonc \
  --device mps --no-wandb --output-dir /tmp/gauge_fields_oriented_slab_smoke_mps_test4

.venv/bin/python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_rank_adaptive_metric_smoke_32_2f_32el.jsonc \
  --device mps --no-wandb --output-dir /tmp/gauge_fields_rank_adaptive_metric_smoke_mps_test4
```

## Caveats

This is still not a novel-view proof. The implementation makes the support law
swappable under the same persistent material-field baseline. The next real
question is whether `oriented_slab` or `rank_adaptive_metric` improves view
stress, X-map occupancy, and cheat-probe behavior after matched training, not
whether they slightly improve one-step RGB smoke metrics.

## Matched support-mode benchmarks

Ran matched local MPS, no-W&B benchmarks with the 16-frame, 128px, 2048-element
motion configs. The compared runs used the same data, loss, schedule, element
count, basis count, initial radius, and initial alpha logit; only
`model.support_mode` changed.

Artifacts:

- `outputs/gauge_fields/support_mode_benchmark_80step/summary.md`
- `outputs/gauge_fields/support_mode_benchmark_250step/summary.md`

250-step result:

| support_mode | eval_psnr | eval_l1 | alpha_cov_050 | alpha_cov_090 | coverage_budget | radius_p95 | anisotropy_p95 | motion_delta_mean | xmap_occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `oriented_slab` | 20.6312 | 0.0554 | 0.9885 | 0.7921 | 3.0188 | 3.9842 | 4.3043 | 0.0528 | 0.2910 |
| `screen_disk` | 20.3144 | 0.0582 | 0.9891 | 0.7720 | 2.8528 | 3.1631 | 1.0000 | 0.0647 | 0.2544 |
| `rank_adaptive_metric` | 19.8992 | 0.0606 | 0.9935 | 0.8788 | 3.6851 | 4.4907 | 3.6372 | 0.0394 | 0.2078 |

Decision from these first matched runs:

- `oriented_slab` is the best candidate to carry forward. It wins source-view
  PSNR/L1 and has the highest X-map occupancy without requiring the most
  coverage.
- `screen_disk` remains the stable control. It is close enough that any stronger
  claim needs view stress or held-out-camera evidence.
- `rank_adaptive_metric` should not be promoted yet. It has the highest
  high-alpha coverage and largest coverage budget, but worse PSNR/L1 and lower
  X-map occupancy. That looks more like broad support / smear than better
  material identity. The next fair version needs spectrum or rank regularization
  before this can be treated as the intended rank-adaptive primitive.

Architecture calls made without blocking on chief-scientist clarification:

- Keep `J Sigma J^T` as the default support transport law because this is meant
  to be a transported material support, not a per-frame camera-facing kernel.
- Keep `oriented_slab` thin-initialized rather than hard-constrained for the
  first benchmark. If it continues to win, add an explicit thickness or
  surface-rank penalty as a follow-up.
- Treat `rank_adaptive_metric` as a full-rank PSD metric candidate in this first
  implementation. Literal rank adaptivity should be added through eigenvalue
  sparsity / MDL pressure after the baseline comparison, not baked in before we
  can see its failure mode.

## Multi-camera held-out validation lane

The source-view benchmark was not enough. The user correctly pushed that the
scores were close and source-view overfit can select the wrong representation.
Added a `frame_source = "multicam_val"` path to the gauge trainer that:

- loads the generated multi-camera validation manifest,
- trains on `source_frames`,
- evaluates local `heldout_*` metrics on `target_frames`,
- writes `heldout_preview.png` and `heldout_side_by_side.mp4`,
- stores held-out camera tensors in the checkpoint.

For the first reliable calibrated lane, used the DeepView sample
`deepview_03_Dog_camera_0001_to_camera_0015`. The manifest exposes
`models.json`, so the trainer converts DeepView source/target camera models into
a relative source-camera coordinate frame. This is still a pinhole approximation
to DeepView fisheye imagery, so the metric is much better than source-view PSNR
but still not a final dataset-quality number.

New configs:

- `src/train_configs/local_mac_gauge_fields_multicam_deepview_screen_disk_128_16f_2048el.jsonc`
- `src/train_configs/local_mac_gauge_fields_multicam_deepview_oriented_slab_128_16f_2048el.jsonc`
- `src/train_configs/local_mac_gauge_fields_multicam_deepview_rank_adaptive_metric_128_16f_2048el.jsonc`

80-step local MPS benchmark:

| support_mode | source_psnr | heldout_psnr | heldout_l1 | heldout_cov_050 | heldout_cov_090 | heldout_coverage_budget | source_xmap_occ | heldout_xmap_occ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `screen_disk` | 24.6535 | 9.6479 | 0.2402 | 0.7915 | 0.6925 | 7.7138 | 0.1409 | 0.1394 |
| `rank_adaptive_metric` | 24.2903 | 9.5662 | 0.2412 | 0.7839 | 0.7465 | 11.9799 | 0.1682 | 0.1560 |
| `oriented_slab` | 25.0395 | 9.3344 | 0.2478 | 0.7661 | 0.7159 | 10.3640 | 0.2004 | 0.1995 |

This reverses the source-view story. `oriented_slab` wins source PSNR again, but
it is worst on held-out-camera PSNR in this first calibrated lane. `screen_disk`
wins held-out PSNR, and `rank_adaptive_metric` is close while using much more
target-view coverage. The right conclusion is not "throw slab or rank-adaptive
out"; it is "source-view PSNR is not the selector." Keep all three in the
benchmark suite and rank them primarily by calibrated held-out camera metrics
plus cheat probes.

## Direct 3DGS control

Added a direct free-dynamic 3DGS baseline for the same DeepView source/held-out
camera bundle:

- `research_experiments/gauge_fields/train_splat_baseline.py`
- `src/train_configs/local_mac_splat_baseline_multicam_deepview_free_dynamic_3dgs_128_16f_2048splats.jsonc`

This baseline uses per-frame learnable Gaussian splats with no video encoder and
no implicit-camera model. It shares the same source frames, held-out target
frames, relative DeepView camera pose, splat count / element count, initial
depth, and initial scale/radius as the gauge-field comparison. It uses the
repo's dense PyTorch 3DGS renderer, so it is slower and does not expose the same
alpha/X-map/projection diagnostics as the gauge renderer.

Updated 80-step local MPS held-out-camera table:

| representation | source_psnr | heldout_psnr | heldout_l1 |
| --- | ---: | ---: | ---: |
| `free_dynamic_3dgs` | 20.5017 | 9.7392 | 0.2357 |
| `screen_disk` | 24.6535 | 9.6479 | 0.2402 |
| `rank_adaptive_metric` | 24.2903 | 9.5662 | 0.2412 |
| `oriented_slab` | 25.0395 | 9.3344 | 0.2478 |

This makes the source/held-out mismatch even sharper: the direct 3DGS control
has much worse source-view fit at 80 steps than every gauge mode, but the best
held-out-camera PSNR/L1 in this first lane. Do not interpret the tiny held-out
margin as a decisive win, but do treat it as evidence that source-view overfit
quality is not a reliable representation selector.
