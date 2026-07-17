# Gauged UVT Bq4 Fresh-Process Median Gate

## Context

The first fresh-process isolation pass used one repeat and proved only that
process isolation reduced the worst warmed feature-state-update spike. It was
not strong enough to be a timing acceptance rule. This pass upgrades the same
report to include median ratios and a warmup-discard view, then reruns the saved
artifact with three repeats.

## Code Changes

Report:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_fresh_process_report.py
```

The report now records:

- median no-first, projective-total, and feature-state-update ratios;
- `warmup_discard_repeats`;
- `acceptance_ratio_threshold`;
- a `timing_acceptance` block with post-warmup pairs, post-warmup summary, and
  `status` equal to `pass`, `fail`, or `insufficient`.

Focused test:

```text
tests/test_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process_report.py
```

The test now checks that warmup-discard can reject a bad first repeat while
accepting the post-warmup median behavior.

## Artifact

Command:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_fresh_process_report.py \
  --repeats 3 \
  --warmup-discard-repeats 1 \
  --out-dir outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process
```

Saved artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
```

## Result

Core contract:

- `status = ok`
- `requested_repeat_count = 3`
- `warmup_discard_repeats = 1`
- `paired_ratio_count = 6`
- all rows are fresh processes
- all expected global steps traced
- all projective interval substeps present
- cache/support clean

All-pair summary:

```text
no_first_bump_count = 0
projective_total_bump_count = 1
feature_state_update_bump_count = 2
max_no_first_ratio = 0.7087283466117477
max_projective_total_ratio = 2.2454207580524894
max_feature_state_update_ratio = 1.2948922914387324
median_no_first_ratio = 0.6530516888499702
median_projective_total_ratio = 0.8356591487478802
median_feature_state_update_ratio = 0.7124745747568637
```

Post-warmup acceptance:

```text
status = pass
post_warmup_pair_count = 4
median_ratios_within_threshold = true
median_no_first_ratio = 0.5645123618278631
median_projective_total_ratio = 0.8356591487478802
median_feature_state_update_ratio = 0.846418513757801
max_projective_total_ratio = 2.2454207580524894
max_feature_state_update_ratio = 1.2948922914387324
```

## Interpretation

This is a cleaner timing story than the warmed in-process policy-order report:
the measured policy has no no-first bump under fresh-process repeats, and the
post-warmup median ratios are all below 1.0. But there is still one large
post-warmup projective-total outlier and two feature-state-update max-ratio
bumps across all six pairs.

Current belief:

- use fresh-process median plus warmup-discard as the timing acceptance view;
- keep max-ratio substep outliers as caveats and profiling targets;
- do not change the fiber/gauge math based on these spikes, because
  cache/support and traced-step contracts are clean.

## Validation

Artifact verifier:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_fresh_process_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process/summary.json
```

passed.

Focused test:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process_report.py -q
```

passed: `6 passed in 4.35s`.

Expanded focused report pack:

```text
142 passed in 8.17s
```
