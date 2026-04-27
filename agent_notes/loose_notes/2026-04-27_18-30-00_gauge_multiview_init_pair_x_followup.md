# Gauge Multi-View Init and Pair-X Followup

Date: 2026-04-27

## Motivation

The first 3-camera result showed that the naked gauge field lost badly to the free dynamic 3DGS baseline:

| model | heldout PSNR |
| --- | ---: |
| `free_dynamic_3dgs` | `13.2940` |
| `rank_adaptive_metric/projected_conic` | `7.7890` |
| `screen_disk/projected_conic` | `7.4607` |

The two most direct missing pieces from the research handoff were:

- initialize material points from both train cameras, not only the anchor camera
- add a pairwise canonical-coordinate consistency loss over rendered `X` maps

## What Changed

Added multi-view first-frame initialization:

```text
model.init_source = anchor_first_frame | multiview_first_frames
```

`multiview_first_frames` splits the material points across the configured train cameras, backprojects first-frame image grids at `init_depth`, and transforms non-anchor samples into the anchor coordinate frame using the calibrated train `w2c`.

Added pairwise `X` consistency:

```text
losses.pair_x_weight
losses.pair_x_depth_sigma
losses.pair_x_alpha_min
losses.pair_x_every
```

The loss renders two train cameras at the same frame, reconstructs source world positions from source rendered depth, projects those points into the destination camera, samples destination `X`, depth, and alpha, and penalizes disagreement between source `X` and destination `X` under an alpha/depth visibility mask.

This is a practical approximation of the pairwise sheaf idea:

```text
same rendered 3D point -> same canonical material coordinate
```

## Files

- `research_experiments/gauge_fields/data.py`
- `research_experiments/gauge_fields/train.py`
- `research_experiments/gauge_fields/run_deepview_3cam_holdout.py`
- `src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_smoke_32_2f_64el.jsonc`
- `src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_128_16f_2048el.jsonc`

## Verification

Compile:

```bash
uv run python -m py_compile \
  research_experiments/gauge_fields/data.py \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/run_deepview_3cam_holdout.py
```

Tests:

```bash
uv run --with pytest python -m pytest tests/test_gauge_incidence.py
```

Result: `4 passed`.

Smoke:

```bash
uv run python research_experiments/gauge_fields/train.py \
  src/train_configs/local_mac_gauge_fields_multicam_deepview_3cam_train2_test1_rank_adaptive_metric_multiview_init_pair_x_smoke_32_2f_64el.jsonc \
  --device mps \
  --steps 2 \
  --no-wandb \
  --output-dir /tmp/gauge_multiview_init_pair_x_smoke
```

Result:

- `pair_x` logged nonzero at both smoke steps
- `heldout_eval_psnr`: `6.4924`

Full 80-step run:

```bash
uv run python research_experiments/gauge_fields/run_deepview_3cam_holdout.py \
  --steps 80 \
  --device mps \
  --no-wandb \
  --only rank_adaptive_metric_multiview_init_pair_x \
  --output-root outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes
```

Result:

- `eval_psnr`: `19.7964`
- `heldout_eval_psnr`: `8.1434`
- wall clock: `971.12s`

Updated summary:

```bash
uv run python research_experiments/gauge_fields/summarize_runs.py \
  outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes \
  --sort-by heldout_eval_psnr \
  --out-md outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes/summary.md \
  --out-json outputs/gauge_fields/multicam_deepview_3cam_train2_test1_80step_fast_modes/summary.json
```

## Comparison

Same 128px / 16f / 2048 primitives / 80-step budget:

| model | heldout PSNR | wall clock |
| --- | ---: | ---: |
| `free_dynamic_3dgs` | `13.2940` | `184.31s` |
| `rank_adaptive_metric + multiview_init + pair_x` | `8.1434` | `971.12s` |
| `rank_adaptive_metric/projected_conic` | `7.7890` | `478.65s` |
| `screen_disk/projected_conic` | `7.4607` | `206.51s` |

## Takeaway

The intended gauge collaborators helped, but only modestly:

```text
7.7890 -> 8.1434 heldout PSNR
```

That confirms the direction is not dead math, but it is nowhere close to the free 3DGS control in this harness. The added pairwise loss is also expensive in pure Torch because it adds extra full renders.

The next useful steps are not more weight guessing on this exact loop. Better candidates:

- add a faster/fused renderer path before making pairwise losses heavier
- test a constrained splat baseline with the same low-rank/persistent limits
- add real depth or track supervision if we want gauges to win on geometry
- report gauge wins, if any, on certificates/stress metrics rather than PSNR alone
