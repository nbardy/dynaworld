# Gauge Incidence Matrix 80-Step Run

Date: 2026-04-27

## Context

After adding `render.incidence_mode`, we ran the first full-size DeepView
held-out-camera matrix at the same scale as the recent support-mode benchmark:

```text
128px / 16 frames / 2048 elements-or-splats / 80 steps / MPS / no W&B
```

The output root is:

```text
outputs/gauge_fields/multicam_deepview_incidence_matrix_80step/
```

Each run writes:

```text
metrics.json
wall_clock.json
preview.png
side_by_side.mp4
heldout_preview.png
heldout_side_by_side.mp4
checkpoint.pt
```

## Runner

Added:

```text
research_experiments/gauge_fields/run_deepview_incidence_matrix.py
```

Command:

```bash
uv run python research_experiments/gauge_fields/run_deepview_incidence_matrix.py \
  --steps 80 \
  --device mps \
  --no-wandb \
  --output-root outputs/gauge_fields/multicam_deepview_incidence_matrix_80step
```

Summary command:

```bash
uv run python research_experiments/gauge_fields/summarize_runs.py \
  'outputs/gauge_fields/multicam_deepview_incidence_matrix_80step/*' \
  --sort-by heldout_eval_psnr \
  --out-md outputs/gauge_fields/multicam_deepview_incidence_matrix_80step/summary.md \
  --out-json outputs/gauge_fields/multicam_deepview_incidence_matrix_80step/summary.json
```

`summarize_runs.py` now reads optional `wall_clock.json` and reports
`wall_clock_sec` / `wall_clock_min`.

## Results

| run | eval_psnr | heldout_eval_psnr | heldout_eval_l1 | wall_clock_min | heldout_xmap_occ | heldout_projection_coverage_budget |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| rank_adaptive_metric / ray_gaussian_line_peak | 21.6338 | 11.5705 | 0.1907 | 16.8116 | 0.0664 | 35.3015 |
| rank_adaptive_metric / ray_gaussian_line_mass | 24.3293 | 9.9005 | 0.2327 | 26.9624 | 0.1372 | 13.1436 |
| free_dynamic_3dgs | 20.5017 | 9.7392 | 0.2357 | 2.1135 | n/a | n/a |
| screen_disk / projected_conic | 24.6535 | 9.6479 | 0.2402 | 1.3610 | 0.1394 | 7.7138 |
| rank_adaptive_metric / projected_conic | 24.2230 | 9.5814 | 0.2406 | 3.1677 | 0.1575 | 11.9125 |

## Interpretation

`ray_gaussian_line_mass` is the clean exact-incidence candidate. It preserved
source-view fit and improved held-out PSNR versus the projected-conic
rank-adaptive metric baseline:

```text
9.9005 vs 9.5814 heldout PSNR
```

but it is very expensive in the current pure Torch all-elements/all-pixels
implementation:

```text
26.96 min vs 3.17 min for rank_adaptive_metric / projected_conic
```

`ray_gaussian_line_peak` produced the best held-out PSNR, but it is not a clean
win yet. It has weaker source fit, much larger projected coverage, low
held-out X-map occupancy, and high held-out projection coverage:

```text
eval_psnr 21.6338
heldout_projection_coverage_budget 35.3015
heldout_xmap_occ 0.0664
```

This may be a broad-coverage / blur-like generalization effect rather than
better geometry. It needs visual inspection and coverage-matched retuning
before treating it as a representation improvement.

## Next

1. Inspect held-out videos/contact sheets for the top two line-integral runs.
2. Add a compact timing/quality selector:

```text
heldout_psnr_per_min = heldout_eval_psnr / wall_clock_min
```

3. Tune the mass-normalized line mode first. It is the cleanest exact-incidence
law.
4. Treat peak-density as a diagnostic/retuning branch unless qualitative
held-out render quality is clearly better.
5. If line incidence stays useful, the next implementation problem is candidate
culling or tiled evaluation. The math is not the bottleneck; the pure Torch
evaluation pattern is.
