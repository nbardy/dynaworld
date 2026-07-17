# Compiled-Adjoint Replacement Gap Close

## Context

The active goal remains:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous completion-gap artifact had all broad quality/media/frame-count and
timing gaps closed, but still held `full_compiled_adjoint_trainer_replacement`
partial. The question was whether the current projective interval trainer route
was genuinely using the compiled interval Metal forward/backward path, or only a
cached forward with ordinary per-frame autograd.

## What Changed

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_compiled_adjoint_replacement_report.py
tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json
```

The new report verifies three layers:

1. Saved broad10 real-video trainer/quality/media behavior.
2. Case payloads proving the trainer used the projective interval runtime path,
   RGB direct-loss autograd, renderer gradient flags, and clean cache/support
   behavior across 20 cadence/measured rows.
3. Source contract proving the autograd bridge is the compiled interval route:
   `_ProjectiveCellIntervalBackward.forward()` calls
   `render_projective_trace_cell_interval_atlas_metal`, and
   `_ProjectiveCellIntervalBackward.backward()` calls
   `direct_backward_projective_trace_cell_interval_atlas_metal`.

Important scope label: this is the practical direct-atomic RGB trainer route
backed by the compiled interval Metal adjoint. It is not deterministic compact
static-STAR promotion.

## Evidence

The replacement report verifies:

```text
final_compiled_adjoint_replacement_accepted = true
compiled_trainer_replacement_gap = 0
case_payload_count = 20
all_cases_projective_interval_main_path = true
all_cases_rgb_direct_loss = true
all_cases_gradient_flags_present = true
all_cases_backward_timing_present = true
measured_cache_reuse_ok = true
source_contract_checks_pass = true
broad_frame_count_count = 4
broad10_trainer_distinct_youtube_id_count = 10
broad10_quality_distinct_youtube_id_count = 10
broad10_media_distinct_youtube_id_count = 10
```

Regenerated:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

The gap report now records:

```text
open_gap_ids = ["full_goal_completion"]
proved_requirement_count = 5
partial_requirement_count = 1
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

The last two fields stay false/true because this artifact is an evidence-gap
audit, not the top-level final completion audit. The separate goal-progress
artifact still keeps `full_goal_completion` open.

## Verification

Commands run:

```text
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_real_video_compiled_adjoint_replacement_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_compiled_adjoint_replacement/summary.json

PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json --verify-current-inputs

PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py -q

PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_goal_completion_gap_report.py tests/test_star_uvt_projective_real_video_compiled_adjoint_replacement_report.py -q
```

Results:

```text
compiled-adjoint report verified
completion-gap report verified against current inputs
8 passed in 6.39s
23 passed in 4.45s
```

## Decision Implication

The old completion-gap blocker is closed. The remaining honest blocker is no
longer a specific missing broad10/trainer/timing artifact; it is the top-level
completion decision/audit for the whole active goal. Future agents should not
reopen `full_compiled_adjoint_trainer_replacement` unless the source contract,
case payloads, or broad10/timing/shared-work evidence regress.

## Update: Goal-Progress Integration

After wiring the compiled-adjoint replacement report into the top-level
goal-progress audit, the saved progress artifact now records
`proved_requirement_count=34` while keeping `full_goal_completion` open. The
regenerated completion-gap report correspondingly records zero concrete
evidence gaps, but its `open_gap_ids` is `["full_goal_completion"]` because the
final completion audit is still intentionally separate from this gap-row
contract.
