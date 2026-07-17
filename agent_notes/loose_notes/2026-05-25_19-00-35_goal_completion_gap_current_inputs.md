# Goal Completion Gap Current-Input Acceptance

## Context

The Gauged UVT Trace Atlas goal remains active: compile 4D spacetime primitives
through a known camera program into reusable sensor-time traces, with clean
derivatives and maximal forward/backward reuse across time. The immediate
problem was not new math; it was audit drift. The goal-completion gap report
could be internally valid while depending on a stale goal-progress artifact.

## What Changed

`research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py`
now validates its goal-progress input through
`verify_projective_goal_progress_current_acceptance(...)`, not only through the
goal-progress report's internal schema checks. The report also exposes
`--verify-current-inputs`, which compares a saved gap artifact against a report
freshly rebuilt from the current default inputs.

The saved artifact remains intentionally non-complete:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json
```

It keeps `completion_ready=false` and `does_not_prove_completion=true`, proves
only `formal_goal_memory_and_audit` and `sublinear_world_side_work_proxy`, and
keeps `broad_real_scene_quality_acceptance`,
`full_compiled_adjoint_trainer_replacement`, and
`timing_acceptance_protocol` partial. The current artifact also checks the
broad10 real-video trainer matrix, so `compiled_trainer_source_gap=0` while
the quality/media, frame-count, strict-timing, and full-trainer-replacement
gaps remain open.

## Evidence

Current-input verifier:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_goal_completion_gap_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_completion_gap/summary.json \
  --verify-current-inputs
```

Result: verified against current inputs.

Focused tests:

```text
tests/test_star_uvt_projective_goal_completion_gap_report.py
12 passed in 3.61s
```

Combined drift-sensitive tests:

```text
tests/test_star_uvt_projective_goal_completion_gap_report.py
tests/test_star_uvt_projective_goal_progress_audit.py
55 passed in 7.10s
```

## Current Model

The gap report is now an executable memory contract. It says:

1. The math/prototype evidence is real enough to preserve as progress.
2. The active goal is still not closed.
3. A future thread cannot close the goal by pointing at stale top-level
   progress evidence.

## Remaining Gaps

- Broad real-scene quality/media acceptance still needs the 10-source coverage
  now shown by the trainer matrix, not only the current five-source tethers.
- Broad frame-count acceptance still needs at least 4 real-video frame-count
  points.
- Strict timing still has two preserved warm-state timing misses, even though
  Bq4 fresh-process post-warmup medians pass.
- Full compiled-adjoint trainer replacement still needs the broad10 trainer
  source coverage extended into the same frame-count, quality, and media
  envelope as the final acceptance gate.

## Decision Implication

Future work should use the completion-gap report as the first "are we done?"
gate. If it still reports partial rows, continue the theory/prototype/testing
loop. If a future artifact claims completion, it must make this report pass
without weakening the targets.
