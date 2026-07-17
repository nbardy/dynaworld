# Frame-Count Breadth Gap Close

Date: 2026-05-25

## Goal Anchors

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Context

The broad10 media tether closed the source/media evidence gaps, but the
completion-gap report still had `broad_quality_frame_count_gap=1`. Existing
frame-count coverage in the main source-distinct matrix was `4,8,16`, so the
acceptance envelope needed one more real-video frame-count point without
pretending that strict timing had passed.

## What Changed

Ran a 4-count multiscene frame-scaling matrix over source-distinct real-video
segments:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_frame_scaling_matrix_4count/summary.json
```

The raw source artifact intentionally remains `failed` because strict no-first
timing does not beat cadence. The useful facts are separate:

```text
frame_counts = [4, 8, 16, 32]
frame_count_count = 4
frame_growth_factor = 8.0
row_count = 24
all_rows_pass = true
all_rows_no_overflow = true
all_rows_fallback_free = true
all_rows_visibility_stratification_free = true
all_measured_loss_matches_cadence = true
max_measured_vs_cadence_rebuild_ratio = 0.5
max_measured_support_rebins = 0
max_measured_stale_refreshes = 0
max_measured_no_first_growth_vs_frame_growth_ratio = 0.22855493152192446
max_measured_vs_cadence_no_first_step_ms_ratio = 1.3397420089864893
```

Added a diagnostic artifact that accepts frame-count breadth only when the
source failure is exactly the expected strict timing failure:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic/summary.json
```

Its important flags:

```text
status = ok
frame_count_breadth_accepted = true
strict_failed_only_expected_timing = true
no_first_timing_win = false
no_first_growth_sublinear = true
```

Then wired that diagnostic into:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_acceptance_envelope_report.py
research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py
tests/test_star_uvt_projective_real_video_frame_count_breadth_diagnostic_report.py
tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py
tests/test_star_uvt_projective_goal_completion_gap_report.py
```

## Current Gap State

After regenerating and verifying the acceptance envelope, goal-progress audit,
and completion-gap report, the completion-gap summary is:

```text
completion_ready = false
broad_quality_source_gap = 0
broad_media_source_gap = 0
broad_quality_frame_count_gap = 0
compiled_trainer_source_gap = 0
strict_timing_failure_gap = 2
```

This closes source/media/frame-count coverage bookkeeping. It does not close
the active goal. Broad real-scene acceptance, full compiled-adjoint trainer
replacement, and the strict timing acceptance protocol remain partial.

## Verification

```text
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json --verify-current-inputs
```

Result:

```text
verified outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json against current inputs
```

Focused regression suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_frame_count_breadth_diagnostic_report.py \
  tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py \
  tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
90 passed in 4.28s
```

## Interpretation

The important hygiene point is that frame-count breadth and timing acceptance
are now disentangled. We have evidence that the projective interval path keeps
loss/cache/support/fallback invariants clean across an `8x` frame-count growth,
but strict timing still has two accepted misses. The next timing decision is
therefore not a fiber-bundle or gauge-math change; it is either to make the
strict warm-state timing gate pass or to promote a stronger fresh-process
median protocol as the final timing contract.
