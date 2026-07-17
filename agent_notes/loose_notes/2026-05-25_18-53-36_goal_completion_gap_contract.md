# Goal Completion Gap Contract

## Context

The active objective is still not complete:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward work over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The goal-progress audit now has 33 proved requirements and one open
`full_goal_completion` row. The problem was that the open row was descriptively
right but not yet machine-checked: future work still had to infer what counts
as "broad" or "full replacement" from prose.

## What Changed

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py
tests/test_star_uvt_projective_goal_completion_gap_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

The report verifies four current evidence inputs:

```text
goal_progress
real_video_acceptance_envelope
real_video_timing_variance_envelope
shared_work
```

It then derives a non-completion gap contract. It proves only:

```text
formal_goal_memory_and_audit
sublinear_world_side_work_proxy
```

and keeps these partial:

```text
broad_real_scene_quality_acceptance
full_compiled_adjoint_trainer_replacement
timing_acceptance_protocol
```

## Concrete Current Gaps

The saved report currently records:

```text
completion_ready = false
does_not_prove_completion = true
broad_quality_source_gap = 5
broad_quality_frame_count_gap = 1
strict_timing_failure_gap = 2
compiled_trainer_source_gap = 5
```

Interpretation:

- broad quality/trainer acceptance targets now stay at least 10 distinct
  sources;
- final real-video frame scaling needs at least 4 frame-count points, while
  current real-video evidence has 3;
- the strict timing gate still has 2 failures even though fresh-process median
  timing passes;
- compiled-adjoint trainer replacement is still focused evidence, not a broad
  full replacement.

This is deliberately not a completion audit. It makes the next completion audit
harder to fake.

## Verification

Commands run:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py \
  tests/test_star_uvt_projective_goal_completion_gap_report.py

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_completion_gap_report.py -q
```

Result:

```text
8 passed in 1.65s
```
