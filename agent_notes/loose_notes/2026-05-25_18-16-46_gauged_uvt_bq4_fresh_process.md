# Gauged UVT Bq4 Fresh-Process Timing Isolation

Supersession note: the same artifact path was later overwritten with a
three-repeat run plus warmup-discard median acceptance. This note preserves the
earlier one-repeat sanity-check reading; see
`agent_notes/loose_notes/2026-05-25_18-28-36_gauged_uvt_bq4_fresh_process_median.md`
for the current saved artifact interpretation.

## Context

The previous Bq4 trace reports narrowed the real-video timing caveat away from
support/candidate math:

- traced spike-step rerun did not reproduce the saved Bq4 no-first miss, but
  showed one 16f projective interval / feature-state-update bump;
- 16f repeat-stability made that bump disappear in a simple repeated schedule;
- sequence-order and warmed policy-order reports showed order/warm-state
  substep variance, with the worst warmed feature-state-update ratio reaching
  `1.9605903379413647`.

This note records the follow-up that isolates each target policy-order case in a
fresh Python/MPS process.

## Artifact

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
```

Report script:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_fresh_process_report.py
```

Verifier test:

```text
tests/test_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process_report.py
```

The saved artifact used `--repeats 1` to keep this heartbeat pass bounded. It
spawns one worker process per target row for the `16f` Bq4 pair in
`cadence_then_measured` and `measured_then_cadence`.

## Result

The verifier confirms:

- every row has `fresh_process = true`;
- all expected global steps are traced;
- every traced chunk has projective interval substep timing;
- cache/support remains clean: zero fallback marks, support rebins, stale
  refreshes, and tile overflow.

Summary:

```text
paired_ratio_count = 2
no_first_bump_count = 1
projective_total_bump_count = 1
feature_state_update_bump_count = 1
max_no_first_ratio = 1.533856565479194
max_projective_total_ratio = 1.1251372453964577
max_feature_state_update_ratio = 1.1289229329016697
```

Per policy order:

- `cadence_then_measured`: no-first ratio `0.8318546740246096`,
  projective-total ratio `1.1251372453964577`, feature-state-update ratio
  `1.1289229329016697`.
- `measured_then_cadence`: no-first ratio `1.533856565479194`,
  projective-total ratio `0.24588494252480544`, feature-state-update ratio
  `0.1841471958493282`.

## Interpretation

Fresh-process isolation does not make timing acceptance clean in one repeat:
there is still one no-first bump and one projective/feature-state bump. It does,
however, shrink the worst warmed in-process feature-state-update ratio from
`1.9605903379413647` to `1.1289229329016697` in this check.

So the current read is:

- the Bq4 caveat is still timing methodology / launch warm-state / process-state
  variance, not a reason to change the fiber/gauge formulation;
- broad timing acceptance should use fresh-process medians, warmup discard, and
  enough repeats before interpreting substep bumps as renderer math failures;
- cache/support remains innocent in this branch.

## Validation

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_fresh_process_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
```

passed.

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process_report.py -q
```

passed: `5 passed in 7.78s`.

The expanded focused pack including the new fresh-process verifier passed:

```text
140 passed in 5.15s
```
