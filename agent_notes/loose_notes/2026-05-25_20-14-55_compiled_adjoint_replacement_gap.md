# 2026-05-25 20:14:55 Compiled-Adjoint Replacement Gap

## Context

The active Gauged UVT goal remains open. Before this chunk, the completion-gap
report had already closed broad quality, broad media, frame-count breadth, and
timing-protocol gaps. The remaining concrete row was
`full_compiled_adjoint_trainer_replacement`.

## What Changed

Validated and wired:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json
```

The artifact verifies the current practical trainer replacement path:

```text
trainer selects _render_projective_interval_feature_tubes_autograd
trainer harness uses _ProjectiveCellIntervalBackward
forward calls render_projective_trace_cell_interval_atlas_metal
backward calls direct_backward_projective_trace_cell_interval_atlas_metal
visibility order and tile membership are compiled atlas constants
```

It also checks 20 broad10 case payloads:

```text
all projective-interval main path
all RGB direct-loss autograd
all renderer gradient flags present
forward/backward timing present
measured cache reuse ok
zero fallback/support churn
10 broad trainer sources
10 broad quality/media sources
four frame-count points
shared-work ratios below threshold
```

The summary records:

```text
final_compiled_adjoint_replacement_accepted = true
compiled_trainer_replacement_gap = 0
```

## Gap Report Result

The current goal-completion gap now verifies the replacement artifact and
reports:

```text
proved rows = 5
partial rows = 1
open_gap_ids = ["full_goal_completion"]
broad_quality_source_gap = 0
broad_media_source_gap = 0
broad_quality_frame_count_gap = 0
strict_timing_failure_gap = 0
timing_acceptance_gap = 0
compiled_trainer_source_gap = 0
compiled_trainer_replacement_gap = 0
completion_ready = false
does_not_prove_completion = true
```

## Interpretation

This closes the compiled-adjoint replacement row inside the machine gap
contract. Follow-up note
`agent_notes/loose_notes/2026-05-25_20-33-59_goal_completion_promotion_audit.md`
adds the final promotion artifact that consumes this gap report and records
`is_goal_complete=true`.

## Verification

```text
projective_real_video_compiled_adjoint_replacement_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json
projective_goal_completion_gap_report.py
```

Focused tests after wiring the replacement evidence into the progress-audit
fixture:

```text
tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py
tests/test_star_uvt_projective_goal_completion_gap_report.py
tests/test_star_uvt_projective_goal_progress_audit.py

69 passed in 6.65s
```

Wider cross-report gate:

```text
tests/test_star_uvt_projective_real_video_timing_protocol_acceptance_report.py
tests/test_star_uvt_projective_real_video_frame_count_breadth_diagnostic_report.py
tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py
tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py
tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py
tests/test_star_uvt_projective_goal_completion_gap_report.py
tests/test_star_uvt_projective_goal_progress_audit.py

108 passed in 6.12s
```
