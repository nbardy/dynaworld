# STAR UVT Manual VJP Subphase Profile

Date: 2026-05-19

## Context

The manual hidden64 sparse visual VJP cut full-cell8 target-area step time from
`7526.73ms` to `6413.96ms`, but the row remained far too slow and
quality-negative. I added a standalone profiler to split the manual VJP into
subphases before starting a native Metal fork.

## Artifact

Script:

`research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`

Command:

`PYTHONPATH=src/train .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py --repeat 1 --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_fullstep`

Report:

`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_fullstep.md`

JSON:

`outputs/benchmarks/2026-05-19_star_uvt_sparse_visual_manualvjp_subphase_profile_fullstep.json`

## Result

Full-step profile, 32 chunks:

- sparse render extrapolates to `655.17ms`
- manual hidden64 loss VJP extrapolates to `3943.12ms`
- pixel-id build extrapolates to `207.39ms`
- `conv1_param_feature_grad_ms`: `2041.07ms`, `51.8%` of loss VJP
- `fc1_ms`: `991.74ms`, `25.2%` of loss VJP
- `conv2_param_hidden_grad_ms`: `326.23ms`, `8.3%`
- `target_area_loss_grad_pred_ms`: `182.54ms`, `4.6%`

## Read

The target-area reduction is not the big remaining loss-side bottleneck. The
dominant work is the hidden64 colorizer VJP, especially the first-layer feature
gradient and first hidden-layer matmul. The next gate should fuse hidden64
RGB/loss VJP with sparse-pixel backward or a visibility/prefix path, avoiding
Python-side per-pixel hidden tensors and `grad_feature_values` materialization.
