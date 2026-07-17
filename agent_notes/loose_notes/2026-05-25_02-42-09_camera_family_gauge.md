# Camera-Family Gauge

## Context

The active goal-progress audit still had a camera-family gap. The user had
asked earlier whether the right object was a richer fiber bundle over a screen
fiber with a camera gauge. The single-path bundle gauge reports proved
depth/log-depth fiber invariance for one known orbit path, but not for a local
family of camera programs.

## Change

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_camera_family_gauge_report.py
tests/test_star_uvt_projective_camera_family_gauge_report.py
```

Generated:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_gauge/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_camera_family_gauge/summary.md
```

The report extends the base domain from:

```text
Omega x T
```

to:

```text
Q x Omega x T
```

where `q` is a one-parameter local camera-family coordinate that perturbs the
orbit radius, phase, height, and target. It integrates the same spacetime atom
through ordinary depth and log-depth fiber gauges.

## Evidence

Current saved report:

- max value relative error: `8.42e-14`
- max primitive-gradient relative error: `2.40e-12`
- q-gradient relative error: `1.60e-11`
- q finite-difference relative error: `1.49e-10`
- missing-Jacobian controls stay visibly wrong for values, primitive gradients,
  and q gradients

The top-level progress audit now includes this evidence and proves seven rows,
while still keeping `full_goal_completion` open.

Focused tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_camera_family_gauge_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
23 passed in 37.64s
```

## Decision Implication

The theory now has a concrete one-parameter local camera-family guard:

```text
UVT trace over Q x Omega x T = pi_* Gamma(q)^* world_primitive
```

This does not finish the high-dimensional camera-family renderer. The remaining
work is to compile these local family gauges into a reusable Metal atlas without
exploding dimension, memory, or visibility events.
