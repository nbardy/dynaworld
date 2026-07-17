# Shared-Work Goal Audit Strict Contract

## Context

The active Gauged UVT objective is not only "render something." It asks for
4D spacetime primitives compiled through a known camera program into reusable
sensor-time traces, with clean derivatives and maximal compute, memory
bandwidth, and backward-pass reuse so non-pixel costs grow sublinearly with
frame count.

The shared-work audit is the artifact that most directly maps saved evidence
to that objective. It reads the saved revolving-camera fixed-chart artifact and
the trained high-motion scaling artifacts, verifies their own contracts, then
checks aggregate payload, trace, interval-entry, forward, and backward ratios.

Before this pass, the audit made the right high-level checks but trusted its
summary and some derived ratios too much.

## Current Contract

The audit now requires:

```text
report status == ok
benchmark == star_uvt_projective_shared_work_goal_audit
theory_contract mentions sublinear backward/shared-work objective
orbit.frame_counts strictly increasing
trained[*].frame_counts strictly increasing
orbit/trained paths nonempty and unique where applicable
trained sizes unique
all relevant growth/ratio fields finite and positive
summary == summarize(orbit, trained)
underlying orbit verifier passes
underlying trained high-motion verifiers pass
```

Thresholds:

```text
orbit fixed payload growth <= 1.05
orbit per-frame payload growth >= 4.0
orbit final payload ratio < 0.25
orbit final trace ratio < 0.25
orbit final segment ratio < 0.25
orbit final CPU compile ratio < 0.5
orbit final forward ratio < 0.5
orbit final backward ratio < 0.5
trained final interval-entry ratio < 0.20
trained final trace-count ratio < 0.20
trained final forward ratio < 0.75
trained final backward ratio < 0.25
trained shared interval-entry growth < 2.0
trained per-frame interval-entry growth > 4.0
```

This is still not a full production-quality claim. It is an objective-level
evidence gate for the reusable trace-atlas idea.

## What Changed

Files:

```text
research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py
tests/test_star_uvt_projective_shared_work_goal_audit.py
```

The verifier now rejects stale summaries, nonmonotone frame counts, missing or
vague theory contracts, duplicate trained paths/sizes, nonfinite ratios,
nonpositive ratios, bad orbit trace/segment/CPU/forward ratios, bad trained
trace ratios, and bad trained forward ratios.

New mutation tests cover:

- stale summary after orbit row change
- nonmonotone orbit frame counts
- large trained trace-count ratio
- slow trained forward ratio
- missing objective contract

## Verification

Focused audit tests:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_shared_work_goal_audit.py -q

12 passed in 17.57s
```

Saved audit artifact:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_shared_work_goal_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json

verified outputs/benchmarks/2026-05-25_star_uvt_projective_shared_work_goal_audit/summary.json
```

Objective evidence chain:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_bundle_gauge_invariance_report.py \
  tests/test_star_uvt_projective_bundle_gauge_gradient_report.py \
  tests/test_star_uvt_projective_orbit_fixed_chart_scaling_benchmark.py \
  tests/test_star_uvt_projective_shared_work_goal_audit.py -q

43 passed in 9.73s
```

## Decision Implications

The current evidence chain is stronger:

- fiber gauge value/gradient invariance protects the math of `pi_* Gamma^* rho`
- orbit fixed-chart verifier protects the revolving-camera reuse mechanism
- shared-work audit ties saved artifacts back to sublinear world-side growth and
  backward reuse

The remaining work is still broader than this audit: production-quality
rendering, real high-motion revolving paths, and full Metal training-side
acceptance are not proven by this alone.
