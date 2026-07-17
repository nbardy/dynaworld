# STAR UVT Target-Grid Render-Mode Matrix

Date: 2026-05-19 12:08 Asia/Ho_Chi_Minh

## Goal

Continue the fast feature-shader port plan by checking whether the current STAR
UVT feature renderer modes that looked useful in synthetic/direct rows actually
help the current 64f/512px V-JEPA target-grid plus frozen-probe trainer gate.

The specific ambiguity was important:

- `feature_direct_gradcache_reduce_vec4` had synthetic and RGB-target wins.
- `feature_direct_fixedbin` existed as a trainer render mode, but it was only a
  request/eligibility surface, not a separate fixedbin Metal backward.
- The current keeper objective is target-grid V-JEPA MSE plus hidden64 frozen
  RGB-probe loss from the 1300-step checkpoint, not the older RGB-target
  `FeatureToColor` speed diagnostic.

## Code Changes

- Added
  `research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py`.
  It deep-copies the checked analytic-VJP 5-step config, changes only
  `feature_uvt.render_mode`, writes per-mode temporary configs/logs, runs the
  real trainer, and emits a Markdown/JSON matrix.
- Updated `src/train/train_star_uvt_feature_overfit.py` result rows with:
  - `kernel_backward_mode`
  - `requested_fixedbin_is_direct_atomic_alias`
- Changed result reporting so `feature_direct_fixedbin` no longer reports
  `effective_render_mode=feature_direct_fixedbin` when no fixedbin kernel is
  actually selected. It now reports the actual effective render mode as
  `feature_direct_atomic`.

## Commands

Compile and dry run:

```bash
rtk .venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py

rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py \
  --dry-run \
  --modes feature_direct_atomic,feature_direct_gradcache_reduce_vec4 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix_dry
```

Full matrix:

```bash
rtk env TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
  PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix
```

Repeat-top matrix:

```bash
rtk env TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
  PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py \
  --modes feature_direct_atomic,feature_direct_gradcache_cached_bins,feature_direct_gradcache_reduce_vec4,feature_direct_fixedbin \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix_repeat_top
```

## Results

Full matrix artifact:

- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix.json`

All six rows passed and landed on the same final loss/probe PSNR:

| mode | kernel | no-first step | backward |
| --- | --- | ---: | ---: |
| `feature_direct_atomic` | `direct_atomic` | `1541.6ms` | `653.4ms` |
| `feature_direct_gradcache` | `gradcache` | `1639.7ms` | `699.1ms` |
| `feature_direct_gradcache_cached_bins` | `gradcache_cached_bins` | `1530.1ms` | `653.8ms` |
| `feature_direct_gradcache_reduce` | `gradcache_reduce_feature_grad` | `1580.2ms` | `732.1ms` |
| `feature_direct_gradcache_reduce_vec4` | `gradcache_reduce_feature_grad_vec4` | `1641.3ms` | `715.6ms` |
| `feature_direct_fixedbin` | `direct_atomic` | `1499.7ms` | `637.8ms` |

Repeat-top artifact:

- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix_repeat_top.md`
- `outputs/benchmarks/2026-05-19_star_uvt_targetgrid_render_mode_trainer_matrix_repeat_top.json`

Repeat-top result:

| mode | kernel | no-first step | backward |
| --- | --- | ---: | ---: |
| `feature_direct_atomic` | `direct_atomic` | `1249.0ms` | `545.5ms` |
| `feature_direct_gradcache_cached_bins` | `gradcache_cached_bins` | `1410.9ms` | `633.9ms` |
| `feature_direct_gradcache_reduce_vec4` | `gradcache_reduce_feature_grad_vec4` | `1509.6ms` | `709.1ms` |
| `feature_direct_fixedbin` | `direct_atomic` | `1422.6ms` | `627.8ms` |

## Interpretation

The synthetic reducer wins do not transfer to the current target-grid/frozen
RGB-probe trainer objective. `feature_direct_gradcache_reduce_vec4` remains a
useful diagnostic for RGB-target/no-prenorm speed rows, but it is not the
keeper-objective renderer default.

`feature_direct_fixedbin` is confirmed as an alias/request surface over the
direct-atomic kernel. Any timing difference between `feature_direct_atomic` and
`feature_direct_fixedbin` is launch/order noise, not evidence of a fixedbin
implementation.

The next real shader gate is unchanged but sharper: implement an actual
fixedbin/tile-slot feature-gradient path or a fused/native target-grid VJP path
that changes the actual kernel/work boundary, then re-run this same matrix.

## Docs Updated

- `README.md`
- `PROJECT_INDEX.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`
