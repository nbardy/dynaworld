# Gauged UVT Acceptance Envelope Bq4 Timing Integration

## Context

The Bq4 fresh-process traced timing report had moved the timing story from
"strict real-video timing win" to a more precise claim:

```text
strict timing win: false
fresh-process median timing win: true
cache/support correctness: still clean
```

This session wired that evidence into the real-video acceptance envelope and
the top-level projective goal-progress audit so future agents do not have to
manually reconcile the separate timing artifacts.

## What Changed

The acceptance envelope now includes the Bq4 fresh-process timing report as a
first-class evidence row. Its saved summary records nine underlying reports,
six paired fresh-process ratios, one warmup discard repeat, four post-warmup
pairs, zero no-first bumps, and post-warmup medians below cadence:

```text
median no-first ratio:             0.5645123618278631
median projective-total ratio:     0.8356591487478802
median feature-state-update ratio: 0.846418513757801
```

The top-level goal-progress audit now includes both:

```text
real_video_acceptance_envelope
real_video_timing_variance_envelope
```

as proved requirement rows. The regenerated audit reports:

```text
proved_requirement_count: 33
open_requirement_count:   1
is_goal_complete:         false
```

The open row is still broad real-scene quality acceptance and full trainer
replacement beyond focused probes. That is intentional.

## Important Interpretation

The Bq4 artifacts should not be read as "timing solved forever." They say:

1. The strict five-source timing diagnostic still has two expected timing
   failures.
2. Cache/support/tile workload is clean, so the failures are not atlas churn.
3. Warm-state/order probes can still produce max-ratio outliers.
4. Fresh isolated processes with one warmup discard give median timing below
   cadence and no no-first bumps.

So the right acceptance claim is median fresh-process timing, not worst-case
single-process warm-state timing.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_real_video_acceptance_envelope_report.py \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --out-dir outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json \
  --verify-current-inputs

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py \
  tests/test_star_uvt_projective_real_video_timing_variance_envelope_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_real_video_multiscene_bq4_trace_fresh_process_report.py -q
```

Result:

```text
69 passed in 6.27s
```

## Next

Do not change the fiber/gauge math because of the remaining timing outliers.
The current evidence points at process/warm-state MPS variance and feature-state
update/render-forward phase cost, not at a failed camera-ray bundle model.
