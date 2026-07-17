# Projective Goal Progress Audit

## Context

The active objective is broader than any single report: fast 2D rasters across
time from 4D spacetime primitives, with clean derivatives and shared
projection/support/binning/visibility/backward work. The immediate risk was
that the strong local artifacts could be mistaken for full completion.

## Change

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
tests/test_star_uvt_projective_goal_progress_audit.py
```

Generated:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.md
```

The audit now verifies four current evidence bundles:

- bundle gauge invariance
- bundle gauge gradients
- one-parameter local camera-family gauge
- projective shared-work goal audit

Then it maps those to active-goal requirements.

## Current Requirement Rows

Proved:

- formal camera-path compiler contract
- fiber-gauge trace invariant
- clean fiber derivatives
- one-parameter local camera-family bundle math over `Q x Omega x T`
- Metal time-shared forward/backward
- finite-exposure / rolling-shutter fallback
- sublinear world-side work proxy

Open:

- full goal completion

The audit deliberately requires `status = "in_progress"` and
`is_goal_complete = false`. It rejects a report that tries to claim completion
without proving the remaining rows.

## Remaining Gaps

The open row lists these gaps:

- focused artifacts are not yet broad real-scene quality acceptance
- local VJPs/fallbacks are not yet a full compiled-adjoint trainer replacement
- high-dimensional camera-family compilation and Metal atlas reuse are beyond
  the current one-parameter CPU camera-family gauge

## Evidence

The saved goal-progress artifact verifies by CLI.

Focused test command:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_shared_work_goal_audit.py \
  tests/test_star_uvt_projective_bundle_gauge_invariance_report.py \
  tests/test_star_uvt_projective_bundle_gauge_gradient_report.py -q
```

Result:

```text
23 passed in 37.64s  # camera-family + goal-progress focused tests
```

## Decision Implication

Use this audit when answering "are we done?" The correct current answer is:

```text
Several core mathematical and Metal/shared-work requirements are proved on
focused artifacts; the full objective is still active because broad acceptance,
full compiled-adjoint training, and high-dimensional camera-family Metal reuse
remain open.
```
