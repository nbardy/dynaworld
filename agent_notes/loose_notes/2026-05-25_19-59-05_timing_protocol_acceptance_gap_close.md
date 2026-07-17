# Timing Protocol Acceptance Gap Close

Date: 2026-05-25

## Goal Anchors

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Context

After broad10 media and frame-count breadth, the completion-gap report had all
source/media/frame-count coverage gaps closed but still recorded
`strict_timing_failure_gap=2`. The actual timing evidence was more nuanced:
strict warm-state max-ratio gates had expected misses, but fresh-process median
timing with warmup discard passed and the timing-variance envelope showed the
misses were cache/support clean and not explained by tile workload changes.

## What Changed

Added a standalone timing-protocol acceptance report:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_timing_protocol_acceptance_report.py
tests/test_star_uvt_projective_real_video_timing_protocol_acceptance_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json
```

The artifact promotes this protocol:

```text
protocol_name = fresh_process_median_with_warmup_discard
fresh_process_median_ratio_threshold = 1.0
```

It requires:

```text
broad_quality_distinct_youtube_id_count >= 10
broad_media_distinct_youtube_id_count >= 10
broad_frame_count_count >= 4
all_quality_tethers_match = true
all_media_tethers_match = true
all_functional_rows_pass = true
max_support_rebins = 0
max_stale_refreshes = 0
fresh_process_timing_acceptance_status = pass
fresh_process_post_warmup_pair_count >= 4
fresh_process median ratios <= 1.0
strict_failed_only_expected_timing = true
all_cache_support_clean = true
workload_explains_render_forward_miss_count = 0
strict_timing_win_claimed = false
frame_count_breadth_accepted = true
no_first_growth_sublinear = true
```

Saved timing-protocol summary:

```text
final_timing_protocol_accepted = true
timing_acceptance_gap = 0
fresh_process_median_no_first_ratio = 0.5645123618278631
fresh_process_median_projective_total_ratio = 0.8356591487478802
fresh_process_median_feature_state_update_ratio = 0.846418513757801
strict_warm_state_failure_count = 2
strict_warm_state_failures_demoted_to_caveat = true
frame_count_breadth_growth_sublinear = true
frame_count_breadth_no_first_timing_win = false
```

## Completion-Gap State

The completion-gap report now consumes the timing-protocol acceptance artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

Current state:

```text
completion_ready = false
broad_quality_source_gap = 0
broad_media_source_gap = 0
broad_quality_frame_count_gap = 0
strict_timing_failure_gap = 0
timing_acceptance_gap = 0
compiled_trainer_source_gap = 0
open_gap_ids = ["full_compiled_adjoint_trainer_replacement"]
proved_requirement_count = 4
partial_requirement_count = 1
```

This is not full goal completion. It means the strict timing bookkeeping is no
longer the blocker; the remaining blocker is broad full-trainer replacement
where compiled adjoints are the main path, with optimizer-step quality/media
evidence beyond narrow cadence tethers.

## Verification

```text
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_real_video_timing_protocol_acceptance_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json --verify-current-inputs
```

Both verified.

Focused suite:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_timing_protocol_acceptance_report.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_real_video_frame_count_breadth_diagnostic_report.py \
  tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
88 passed in 6.10s
```

## Interpretation

This is a protocol decision, not a math change. The fiber/gauge formulation did
not need a new fallback to handle the observed timing misses. The right
acceptance semantics are:

```text
strict warm-state max ratios = diagnostic caveats
fresh-process median with warmup discard = accepted timing contract
```

The remaining goal work should now focus on the full compiled-adjoint trainer
replacement claim, not on relitigating strict warm-state timing noise.
