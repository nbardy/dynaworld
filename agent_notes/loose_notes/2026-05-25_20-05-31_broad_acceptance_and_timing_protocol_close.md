# 2026-05-25 20:05:31 Broad Acceptance And Timing Protocol Close

## Context

This continues the Gauged UVT / STAR UVT trace-atlas goal:

```text
4D spacetime primitives compiled through a known camera program into reusable
sensor-time traces for fast rasterization across time.
```

The memory anchors remain:

```text
goal: fast 2D rasters across time from 4D spacetime primitives
meta-goal: share projection/support/binning/visibility/backward work over time
key math: UVT trace = pi_* Gamma^* world_primitive
theory: STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous broad10 media note was correct at the time it was written, but the
repo now has additional evidence: frame-count breadth and timing-protocol
acceptance were wired into the completion-gap contract.

## Current Machine State

The current completion-gap artifact is:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

It verifies current inputs and reports:

```text
completion_ready = false
broad_quality_source_gap = 0
broad_media_source_gap = 0
broad_quality_frame_count_gap = 0
strict_timing_failure_gap = 0
timing_acceptance_gap = 0
compiled_trainer_source_gap = 0
open_gap_ids = ["full_compiled_adjoint_trainer_replacement"]
```

The proved rows are:

```text
formal_goal_memory_and_audit
sublinear_world_side_work_proxy
broad_real_scene_quality_acceptance
timing_acceptance_protocol
```

The only partial row is:

```text
full_compiled_adjoint_trainer_replacement
```

## Evidence Added Since Broad10 Media

Frame-count breadth diagnostic:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_frame_count_breadth_diagnostic/summary.json
```

It covers 4/8/16/32-frame evidence, accepts the breadth contract as diagnostic
rather than a strict timing win, keeps support rebins/stale refreshes at zero,
and records sublinear no-first growth under the diagnostic contract.

Timing-protocol acceptance:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_timing_protocol_acceptance/summary.json
```

It promotes fresh-process median timing with warmup discard as the accepted
timing protocol:

```text
fresh_process_median_no_first_ratio = 0.5645123618278631
fresh_process_median_projective_total_ratio = 0.8356591487478802
fresh_process_median_feature_state_update_ratio = 0.846418513757801
timing_acceptance_gap = 0
```

The old strict warm-state failures are not erased. They remain diagnostic
caveats:

```text
strict_warm_state_failure_count = 2
strict_warm_state_failures_demoted_to_caveat = true
strict_timing_win_claimed = false
```

## Interpretation

This does not complete the all-night goal. It changes what is open.

Closed by machine-checked artifacts:

```text
broad10 trainer source coverage
broad10 quality source coverage
broad10 media source coverage
four-count frame breadth
accepted timing protocol
sublinear world-side work proxy
```

Still open:

```text
broad full compiled-adjoint trainer replacement evidence
```

The remaining work should stop spending effort on broad source-count media or
strict warm-state timing as if they are blockers. The next useful benchmark is
a broad trainer-replacement run where compiled adjoints are the main path, with
optimizer-step quality/media outputs inside the same acceptance envelope.

## Verification

The current artifacts verified through:

```text
projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
projective_goal_completion_gap_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json --verify-current-inputs
```

Focused test target to preserve this state:

```text
tests/test_star_uvt_projective_real_video_timing_protocol_acceptance_report.py
tests/test_star_uvt_projective_real_video_frame_count_breadth_diagnostic_report.py
tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py
tests/test_star_uvt_projective_goal_completion_gap_report.py
tests/test_star_uvt_projective_goal_progress_audit.py
```
