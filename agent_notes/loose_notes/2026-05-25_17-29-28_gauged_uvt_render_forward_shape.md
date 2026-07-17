# Gauged UVT Render-Forward Shape Diagnostic

## Context

Continuation of the Gauged UVT Trace Atlas timing investigation. The previous
residual report ruled out saved tile/candidate workload as the explanation for
the Bq4 render-forward miss. This pass asked whether the miss is persistent
per-work-unit latency or a small-number-of-steps timing spike.

## Artifact

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_render_forward_shape_report.py
tests/test_star_uvt_projective_real_video_multiscene_extended_render_forward_shape_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape/summary.json
```

## Findings

- Source remains the failed strict five-source frame-scaling matrix.
- It still fails only the two expected timing gates.
- The report profiles all 15 cadence/measured pairs using saved per-step
  timings.
- `chunk_traces_present_pair_count = 0`, so saved artifacts cannot support
  substep attribution.
- All three no-first timing misses are single-spike driven in render-forward
  time and whole-step time.
- Dropping the largest positive render-forward delta sends the worst no-first
  render ratio to `0.8418254365135661`.
- Max render-forward ratio remains `1.3566329017525305` on
  `Bq4rmeIvJbs_seg_000` at 4f.
- Max no-first render spread is `5.383083741915209`.
- Max no-first render spike delta is `728.0996670015156 ms`.

## Interpretation

The timing miss is now best modeled as a saved per-step timing-shape problem:
identical tile workload, no support/cache/fallback churn, and one positive
render-forward spike dominating each miss. This is not evidence for changing
the atlas math, charting, fibers, or visibility representation. The next
high-signal experiment is a traced Bq4 rerun with `trace_global_steps` around
the spike steps, ideally with render-forward split into interval-cache lookup,
trace evaluation, compositing, and synchronization/driver timing.

## Verification

```text
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_render_forward_shape_report.py
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_extended_render_forward_shape_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_extended_render_forward_shape/summary.json
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_extended_render_forward_shape_report.py -q
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest ...focused multiscene/tether/guarded/audit pack... -q
```

Results:

```text
6 passed in 0.10s
114 passed in 2.21s
```
