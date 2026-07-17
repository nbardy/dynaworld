# Goal Audit Bq4 Cross-Envelope Consistency

## Context

The projective goal-progress audit now depends on two top-level timing
envelopes:

```text
real_video_acceptance_envelope
real_video_timing_variance_envelope
```

Both include the Bq4 fresh-process timing evidence. Before this session, each
envelope was verified independently, but the top-level audit did not prove that
their Bq4 median/status values still agreed with each other.

## Change

Added a cross-artifact verifier in:

```text
research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
```

The top-level verifier now requires the acceptance envelope and timing-variance
envelope to agree on:

```text
fresh-process timing status
fresh-process post-warmup pair count
median no-first ratio
median projective-total ratio
median feature-state-update ratio
```

The top-level summary now also exposes the timing-variance projective-total and
feature-state-update medians plus both strict/fresh-process timing claim flags.

## Why This Matters

This protects the acceptance story from a subtle failure mode: one envelope
could be regenerated from newer Bq4 evidence while the other still points at an
older fresh-process artifact, and both could pass their local contracts. The
goal-progress audit now rejects that drift.

## Verification

Commands run:

```bash
.venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_goal_progress_audit.py

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_progress_audit.py -q

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

Results:

```text
43 passed in 5.04s
goal-progress artifact verified against current inputs
70 passed in 4.76s
```

## Current Interpretation

The top-level timing claim remains deliberately narrow:

```text
strict timing win: false
fresh-process median timing win: true
completion: false
```

The math/gauge model is still not the suspected cause of the remaining
warm-state timing outliers.
