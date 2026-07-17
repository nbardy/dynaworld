# STAR UVT Sparse-Grid Target VJP

Date: 2026-05-19 12:53 +07

## Goal

Continue the sparse-pixel VJP gate by removing the remaining dense
image-gradient materialization and scan. The desired mode is a direct sparse
target-grid/probe VJP for the current 64f/512px/8192-tube STAR UVT
target-grid plus frozen RGB-probe objective.

## Implementation

- Added `feature_target.image_vjp_mode=analytic_sparse_grid`.
- Added a cached trilinear sparse-VJP lattice planner in
  `src/train/train_star_uvt_feature_overfit.py`.
- Added a no-duplicate fast path for the current `2` rendered frames to `1`
  target-grid frame shape, so the hot path avoids unnecessary `index_add_` and
  nonzero filtering.
- Added a device-plan cache for sparse ids/weights, avoiding repeated CPU to
  MPS copies of the same trilinear sparse lattice inside every chunk.
- Extended
  `research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py`
  to profile `analytic_sparse_grid`.
- Added replay config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc`.

## Validation

Small dense-vs-sparse trilinear VJP parity passed on CPU and MPS:

```text
shape=(2,5,16,16), target=(1,5,4,4)
max_abs_err=0.0
```

Compile checks:

```text
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py
```

## Benchmarks

Profile command:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc \
  --image-vjp-mode analytic_sparse_grid \
  --warmup 1 \
  --repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_profile
```

Profile result:

```text
pass=true
bridge total=760.6ms
image_vjp=17.2ms
renderer_backward=62.4ms
param_backward=46.1ms
grad_max_abs=4.60e-08
sparse_pixels=65,536
speedup_vs_autograd_total=1.959x
```

Trainer command:

```text
PYTHONPATH=src/train .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc
```

Trainer result:

```text
pass=true
mean_step=861.7ms
no_first_step=795.3ms
last_step=730.5ms
mean_backward=93.6ms
no_first_backward=88.6ms
feature_target_ms=32.7ms
sparse_pack_ms=0.0ms
mean_sparse_pixel_count=65,536
loss/probe movement matches dense analytic and sparse-pixel rows
wandb/offline-run-20260519_130616-xdzqk682
```

Comparison:

| trainer path | no-first step | no-first backward | note |
| --- | ---: | ---: | --- |
| dense analytic | 1318.0ms | 594.0ms | dense renderer backward |
| sparse pixels | 973.7ms | 254.5ms | dense VJP plus sparse scan |
| sparse grid | 795.3ms | 88.6ms | direct sparse target-grid pack with cached device plan |

## Render-Mode Matrix

After the sparse-grid route passed, I reran the current end-to-end render-mode
matrix against the sparse-grid base config:

```text
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.json
```

All rows pass with matching loss/probe movement and zero overflow. Sorted by
no-first step:

```text
feature_direct_gradcache_reduce_vec4: 730.5ms, backward 78.3ms
feature_direct_gradcache_reduce:      740.7ms, backward 80.2ms
feature_direct_gradcache:             759.4ms, backward 83.9ms
feature_direct_fixedbin:              766.4ms, backward 85.1ms, direct_atomic alias
feature_direct_atomic:                779.3ms, backward 87.4ms
feature_direct_gradcache_cached_bins: 787.8ms, backward 90.8ms
```

This means the selected config's existing `feature_direct_gradcache_reduce_vec4`
render mode remains the best checked render mode under sparse-grid VJP. The
single sparse-grid trainer smoke above was slower (`795.3ms` no-first) than the
same-mode matrix rerun (`730.5ms` no-first), so short-run timing remains
session-sensitive; use the matrix for mode ordering.

## Decision

Promote `analytic_sparse_grid` over `analytic_sparse_pixels` for the current
STAR UVT target-grid/frozen-probe diagnostic. It preserves parity and removes
the dense Torch image-gradient pack scan. Before the sparse-forward follow-up
below, the promoted checked pairing was
`analytic_sparse_grid + feature_direct_gradcache_reduce_vec4`.

This is not yet the final native shader. The target-grid/probe sparse VJP still
runs as Torch-side packing feeding the sparse-pixel Metal backward. The next
speed work should fuse target-grid/probe VJP into a native GPU path or move to a
scalar fixedbin/tile-slot feature-gradient accumulator.

## Sparse Forward Follow-Up

The next bottleneck after sparse-grid VJP was forward rendering: the trainer was
still producing the dense 512px feature image before sampling the target-grid
support. I added a sparse feature forward op in the STAR UVT v0 native bridge:

```text
render_uvt_feature_sparse_pixels_with_bins(...)
feature_target.image_vjp_mode=analytic_sparse_grid_forward
```

The op renders only the target-grid support pixels (`65,536`, `0.390625%` of
dense pixels), returns sparse feature/alpha values plus tile bins, and the
trainer folds those values back into the target grid for feature loss and frozen
RGB-probe loss. Backward reuses the sparse-grid VJP pack and sparse
direct-atomic feature backward.

Sparse forward profile:

```text
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile.json

dense full-image feature render: 515.9ms mean
sparse feature-pixel render:     70.5ms mean
speedup:                         7.322x
max_feature_error:               0.0
max_alpha_error:                 0.0
overflow/unstable:               0 / 0
```

Sparse-forward trainer gate:

```text
config: src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforwardvjp.jsonc
json: outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforwardvjp_64f512_from1300_5step.json
wandb: wandb/offline-run-20260519_132936-rzxcd68q

pass=true
mean_step=560.9ms
no_first_step=492.3ms
last_step=413.7ms
mean_backward=131.8ms
no_first_backward=121.3ms
no_first_render_forward=217.0ms
no_first_feature_target=44.2ms
no_first_rgb_probe_loss=41.9ms
loss=0.886537 -> 0.885009
feature_target_loss=0.632124 -> 0.631692
rgb_probe_loss=0.006360 -> 0.006333
rgb_probe_psnr=21.965 -> 21.984
```

At this point this looked like the fastest checked target-grid/frozen-probe
diagnostic. The scale matrix below keeps the correctness result but downgrades
the timing claim to repeat-sensitive. The older sparse-grid dense-forward row
remains the better backward-only reference (`88.6ms` no-first backward). Next
speed work should fuse the target-grid/probe loss+VJP itself into native GPU
work, or implement a real scalar fixedbin/tile-slot feature-gradient
accumulator.

## Sparse Forward Scale Matrix And Repeat

I added and ran:

```text
research_experiments/star_uvt_feature_tubes/sparse_forward_scale_matrix.py
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_scale_128_256_512.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_scale_128_256_512.json
```

All 128/256/512 rows pass sparse-forward parity and trainer checks with zero
overflow:

| size | dense forward | sparse forward | speedup | no-first step | last step | no-first backward | max tile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 128 | 465.3ms | 378.8ms | 1.228x | 379.2ms | 336.8ms | 114.8ms | 43 |
| 256 | 327.8ms | 191.2ms | 1.715x | 494.2ms | 500.9ms | 154.2ms | 57 |
| 512 | 1039.0ms | 581.1ms | 1.788x | 973.0ms | 1548.8ms | 278.9ms | 63 |

The 512px row was much slower than the first isolated sparse-forward artifact,
so I reran 512px in isolation after the scale matrix:

```text
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile_repeat_after_scale.json
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforwardvjp_64f512_from1300_5step_repeat_after_scale.json
wandb/offline-run-20260519_165330-7jskzds5
```

The repeat passes with dense/sparse forward `1091.4ms -> 584.2ms` (`1.868x`),
mean step `708.0ms`, no-first step `598.2ms`, last step `477.6ms`, and no-first
backward `172.5ms`. This changes the speed claim: sparse-forward remains the
selected diagnostic because it is correct and removes dense forward work, but
timing is run-order/session sensitive. Do not cite the first `492.3ms` no-first
row as a stable floor; cite the current repeat band and keep the next gate aimed
at native target-grid/probe loss+VJP.

I then added a repeat-aware harness:

```text
research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_512_repeat3_timing.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_512_repeat3_timing.json
```

The repeat-3 512px trainer gate passes all rows with the same loss/probe
movement and zero overflow. Summary:

```text
no_first_step_ms:     mean 504.9, min 411.0, max 626.4, stdev 110.3
last_step_ms:         mean 468.8, min 409.3, max 549.9, stdev 72.7
no_first_backward_ms: mean 142.2, min 114.7, max 174.4, stdev 30.1
```

This is the timing comparison surface for the next native target-grid/probe
loss+VJP or fixedbin/tile-slot gate.

## Artifacts

```text
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_report.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_profile.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_grid_vjp_profile.json
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparsegridvjp_64f512_from1300_5step.json
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparsegrid_render_mode_trainer_matrix.json
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile.json
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforwardvjp_64f512_from1300_5step.json
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_scale_128_256_512.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_scale_128_256_512.json
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_512_repeat3_timing.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_512_repeat3_timing.json
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_profile_repeat_after_scale.json
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforwardvjp_64f512_from1300_5step_repeat_after_scale.json
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsegridvjp.jsonc
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforwardvjp.jsonc
```
