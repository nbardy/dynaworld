# Real-Video Timing Variance Envelope

## Context

The active thread goal remains:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous real-video acceptance envelope correctly consolidated functional,
quality, media, support, and fresh-process median timing evidence, but the
five-source extended frame-scaling diagnostic still had two strict timing
misses. The important question was whether those misses meant the UVT atlas
math/cache/support story was wrong, or whether they were warm-state/process
timing variance around otherwise clean cache/support rows.

## Current Model

Current belief: the timing misses are not evidence against the camera-ray
bundle atlas theory. They are better modeled as MPS/process warm-state timing
variance around cache/support-clean rows, with a fresh-process median pass.

Observed facts in
`outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_variance_envelope/summary.json`:

- 9 underlying verifiers pass.
- 5 source scenes and 30 source rows are covered.
- Strict failure count remains 2, and both failures are expected timing misses.
- All timing-miss pairs are cache/support clean.
- Workload changes explain 0 render-forward misses.
- All no-first misses keep tile stats identical and are single-spike driven.
- Dropping the largest render-forward spike gives render-forward ratio
  `0.8418254365135661`.
- The Bq4 traced spike does not reproduce.
- Bq4 fresh-process median acceptance passes with median no-first ratio
  `0.5645123618278631`, median projective-total ratio
  `0.8356591487478802`, and median feature-state-update ratio
  `0.846418513757801`.

This does not prove the broad active objective. It preserves
`strict_timing_win_claimed=false` and `does_not_prove_completion=true`.

## Audit Wiring

The goal-progress audit now treats this as a first-class requirement row:

```text
real_video_timing_variance_envelope
```

The regenerated goal-progress artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
```

now reports:

```text
proved_requirement_count = 33
open_requirement_count   = 1
is_goal_complete         = false
```

The saved artifact verifies against current default inputs:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json \
  --verify-current-inputs
```

## Why This Matters

The theory says STAR UVT should share world-side work over time by compiling
world primitives through the camera-ray bundle into reusable sensor-time traces.
If timing misses were caused by changing active sets, support churn, fallback,
visibility stratification, or cache invalidation, the compiler theory would
need a math/representation change. The timing-variance envelope narrows that:
the misses happen with identical tile workload and clean cache/support state.

So the next math/implementation work should not randomly weaken the fiber-gauge
formulation with vague fallback. The right next branch is more precise timing
instrumentation, launch/warm-state isolation, and larger real-scene acceptance
quality gates.

## Falsification Tests

The variance model weakens if any future rerun shows:

- timing misses with support rebins, stale refreshes, fallback, or visibility
  stratification;
- render-forward misses explained by changed tile stats or active-set
  workload;
- repeated fresh-process median no-first/projective/feature-state ratios above
  cadence after warmup discard;
- a reproduced Bq4 traced spike with stable substep attribution to a specific
  UVT compiler phase.

If that happens, revisit the compiler/cache math. Until then, preserve the
bundle-atlas theory and treat the timing miss as a systems variance caveat.

## Verification

Focused checks after wiring:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_goal_progress_audit.py

PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_real_video_timing_variance_envelope_report.py \
  tests/test_star_uvt_projective_real_video_timing_variance_envelope_report.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_timing_variance_envelope_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
51 passed in 6.34s
```
