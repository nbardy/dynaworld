# Projective Shared-Work Goal Audit

## Context

The active objective asks for more than gauge correctness:

```text
share compute and memory bandwidth and backwards passes maximally across time
for sublinear speed with frame growth
```

We already had saved orbit and trained high-motion artifacts, but the evidence
was spread across separate report formats. This pass added an aggregate audit
that reads those artifacts, verifies their own contracts first, then checks the
cross-artifact ratios that map directly to the active goal.

## Work

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py
tests/test_star_uvt_projective_shared_work_goal_audit.py
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.md
```

The audit consumes:

```text
outputs/benchmarks/2026-05-24_star_uvt_revolving_orbit_fixed_chart_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
```

and calls the underlying verifiers before making aggregate claims.

## Evidence

Saved summary:

```text
orbit fixed payload growth: 1.0x
orbit per-frame payload growth: 8.0x
orbit final fixed/per-frame payload ratio: 0.125
orbit final fixed/per-frame backward ratio: 0.267

trained artifact count: 3
max trained shared interval-entry growth: 1.462x
min trained per-frame interval-entry growth: 9.852x
max trained final shared/per-frame interval-entry ratio: 0.148
max trained final shared/per-frame backward ratio: 0.171
```

Verification:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_shared_work_goal_audit.py -q

6 passed, 1 skipped in 29.15s

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
```

## Interpretation

This is an evidence aggregator, not a new renderer kernel. It is still useful:
future claims about the all-night objective can now point to one fail-closed
artifact that says the current saved projective path satisfies the narrow
shared-work evidence we actually have:

- orbit payload is reused instead of replayed per frame
- trained high-motion interval entries grow sublinearly versus per-frame replay
- backward timing follows the interval-entry story on the saved rows

The larger goal remains active because this is not yet a full-resolution,
general renderer acceptance claim.
