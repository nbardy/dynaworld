# Gauge World-Ball Followup And Same-Count Matrix

## Context

We followed up on the diagnosis that the gauge-field variants were losing to
the 3-camera splat baseline because the gauge primitive was both constrained
and under-supported by its intended collaborator losses.

The requested next move was:

```text
simplify the rank-adaptive gauge
run the comparison matrix
```

## Code Changes

Added a simpler support mode:

```text
transported_world_ball
```

This is a persistent material point with scalar 3D world support:

```text
x_i(t) = x_i^0 + sum_l coeff[t,l] basis[i,l]
Sigma_i = r_i^2 I_3
Sigma_screen = J_pi Sigma_i J_pi^T
```

It differs from:

```text
screen_disk:
  Sigma_screen = scalar pixel radius * I_2

rank_adaptive_metric:
  Sigma_world = J_transport G_i J_transport^T
```

The world-ball mode is deliberately simpler than rank-adaptive metric:

```text
no full PSD metric
no local KNN deformation Jacobian for support
no anisotropic material phase yet
```

Also added:

- `losses.pair_x_start_step`, so pairwise X consistency can be delayed until
  after an RGB warmup.
- lazy support-KNN construction: only `oriented_slab` and
  `rank_adaptive_metric` build support KNN.
- lazy ARAP KNN construction and ARAP evaluation: only when
  `arap_weight > 0`.

The lazy KNN change matters. The first 8192-element screen-disk run appeared
stalled because it was paying KNN/ARAP costs that did not affect that support
mode.

## New Configs

Static splat lower bound:

```text
src/train_configs/local_mac_splat_baseline_multicam_deepview_3cam_train2_test1_static_3dgs_128_16f_2048splats.jsonc
```

Same-primitive-count 2048 gauge configs:

```text
src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_screen_disk_multiview_init_pair_x_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_transported_world_ball_multiview_init_pair_x_128_16f_2048el.jsonc
src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_delayed_128_16f_2048el.jsonc
```

Active-parameter-match attempt configs:

```text
src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_screen_disk_multiview_init_pair_x_128_16f_8192el.jsonc
src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_transported_world_ball_multiview_init_pair_x_128_16f_8192el.jsonc
src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_128_16f_7516el.jsonc
```

## Verification

Passed:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/run_deepview_3cam_holdout.py

uv run --with pytest python -m pytest tests/test_gauge_incidence.py
```

Smoke passed for:

```text
static_3dgs
transported_world_ball_8192_multiview_init_pair_x
screen_disk_8192_multiview_init_pair_x after lazy-KNN patch
```

## Matrix Run

Command:

```bash
uv run python research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  --steps 80 \
  --device mps \
  --no-wandb \
  --only free_dynamic_3dgs,static_3dgs,screen_disk_2048_multiview_init_pair_x,transported_world_ball_2048_multiview_init_pair_x,rank_adaptive_metric_2048_multiview_init_delayed_pair_x \
  --output-root outputs/gauge_fields/multicam_deepview_3cam_train2_test1_same_count_support_matrix_80step
```

Summary:

```text
outputs/gauge_fields/multicam_deepview_3cam_train2_test1_same_count_support_matrix_80step/summary.md
```

Results sorted by held-out camera PSNR:

| method | heldout PSNR | train PSNR | wall clock |
| --- | ---: | ---: | ---: |
| static_3dgs | 13.3309 | 18.2827 | 3.82 min |
| free_dynamic_3dgs | 13.2940 | 16.4423 | 3.81 min |
| rank_adaptive_metric_2048 + delayed pair-X | 8.5339 | 20.1060 | 6.29 min |
| transported_world_ball_2048 + delayed pair-X | 8.1976 | 18.2822 | 2.61 min |
| screen_disk_2048 + delayed pair-X | 8.1180 | 20.0640 | 2.50 min |

## Active-Parameter Match Attempt

We attempted the active-parameter-matched gauge route:

```text
screen_disk: 8192 elements
world_ball: 8192 elements
rank_adaptive_metric: 7516 elements
```

The first 8192 screen-disk full run was stopped after roughly 32 minutes with
no completed output. That is a failed speed gate for the pure Torch renderer.

Partial baseline rows from that output root:

```text
outputs/gauge_fields/multicam_deepview_3cam_train2_test1_fair_support_matrix_80step/summary.md
```

showed:

| method | heldout PSNR | wall clock |
| --- | ---: | ---: |
| static_3dgs | 13.3309 | 3.84 min |
| free_dynamic_3dgs | 13.2940 | 3.97 min |

## Interpretation

The simplified world-ball gauge did what it was supposed to do locally:

```text
transported_world_ball > screen_disk
```

but only slightly:

```text
8.1976 vs 8.1180 heldout PSNR
```

The rank-adaptive gauge still won among the 2048-element gauge modes:

```text
8.5339 heldout PSNR
```

So the result did not support deleting rank-adaptive metric yet.

The larger result is more important:

```text
static 3DGS ~= free dynamic 3DGS >> all gauge modes
```

On this DeepView dog split, a static splat model is already enough to beat the
free dynamic model on held-out camera PSNR. That suggests the benchmark is
mostly measuring static cross-camera coverage and camera/projection agreement,
not dynamic material transport.

## Current Read

The gauge modes are still held back by:

- rough constant-depth multi-view plane initialization,
- no depth or flow supervision,
- no densification/pruning,
- no fused/tiled renderer for active-param-matched budgets,
- likely camera-model mismatch or heldout metric bias toward broad coverage.

The next useful technical move is not another support-mode tweak. It is either:

```text
1. depth/triangulated initialization
2. depth/flow losses
3. fused renderer or lower-cost tiled candidate path for high element counts
4. a dynamic benchmark where static 3DGS is not already near the free-dynamic upper bound
```

## Process Notes

No gauge training processes were left running after the completed same-count
matrix. The 8192 active-param attempt was explicitly stopped because it exceeded
a useful local runtime budget in pure Torch.
