# Gauged UVT Render-Forward Residual Diagnostic

## Context

Continuation of the Gauged UVT Trace Atlas goal. The previous phase-profile
diagnostic narrowed the strict five-source frame-scaling miss to render-forward
timing on `Bq4rmeIvJbs_seg_000`, especially the 4f row. This pass asked a
sharper question: is the render-forward miss explained by more saved
candidate/support work, or by residual per-work-unit latency/timing behavior?

## Artifact

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_render_forward_residual_report.py
tests/test_star_uvt_projective_real_video_multiscene_extended_render_forward_residual_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_residual/summary.json
```

The report reads all 15 cadence/measured case pairs from the failed strict
five-source frame-scaling matrix and compares saved per-step render timings
against saved `tile_stats`.

## Findings

- All 15 cadence/measured saved case pairs have identical `tile_stats`.
- The three no-first timing misses also have identical tile workload.
- `max_tile_stats_abs_delta = 0.0`.
- `workload_explains_render_forward_miss_count = 0`.
- Max render-forward ratio is still `1.3566329017525305` on
  `Bq4rmeIvJbs_seg_000` at 4f.
- Max render-forward-per-clipped-ref ratio is also `1.3566329017525305`.
- All rows remain cache/support clean and loss-tethered.

## Interpretation

This rules out the saved candidate/support distribution as the explanation for
the render-forward miss. The remaining positive signal is residual
render-forward latency per clipped tile-tube reference or timing replay noise.
The next useful diagnostic is Bq4 render-forward substep instrumentation or a
controlled replay that separates interval-cache lookup, trace eval, compositing,
and any synchronization/driver timing.

## Verification

```text
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_render_forward_residual_report.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_render_forward_residual_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_residual/summary.json
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_extended_render_forward_residual_report.py -q
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest ...focused multiscene/tether/guarded/audit pack... -q
```

Results:

```text
6 passed in 0.04s
108 passed in 3.53s
```
