# Bundle Gauge Verifier Strict Contract

## Context

The active Gauged UVT goal has two complementary evidence lanes:

- renderer/work reuse: projective interval atlases, orbit fixed charts, real
  trainer cache reuse
- math/gauge legitimacy: the UVT trace is a fiber pushforward
  `pi_* Gamma^* rho`, independent of local depth coordinate when the
  fiber-measure Jacobian and order certificate are correct

After hardening the real-video and orbit fixed-chart verifiers, the next weak
point was the bundle gauge reports. They already tested the right math, but
their verifiers trusted some summary fields instead of recomputing all
row-level evidence.

## Current Model

For a valid monotone fiber gauge `z = h(r)`:

```text
integral rho(Gamma(y, z)) dz
==
integral rho(Gamma(y, h(r))) |dh/dr| dr
```

and for gradients:

```text
d/dtheta integral rho_theta(Gamma(y, z)) dz
==
d/dtheta integral rho_theta(Gamma(y, h(r))) |dh/dr| dr
```

The missing-Jacobian control is not optional. If omitting `|dh/dr|` does not
fail, the test is not actually protecting the bundle measure.

## What Changed

Files:

```text
research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py
research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_gradient_report.py
```

The value verifier now checks:

- `samples`, `near`, and `far`
- row finiteness for sensor coordinates, all integrals, and all errors
- `abs_error = |log_gauge_integral - depth_integral|`
- `rel_error = abs_error / max(|depth_integral|, 1e-12)`
- missing-Jacobian row error consistency
- summary recomputed from rows and order
- positive monotone log-depth derivative and negative orientation-reversing derivative
- order and summary agreement

The gradient verifier now checks:

- `samples`, `near`, and `far`
- finite/positive gradient norms
- finite-difference `abs_error` and `rel_error` consistency
- summary recomputed from value fields, rows, and finite-difference data
- missing-Jacobian value and gradient controls
- stale summary fields

Tests added:

```text
tests/test_star_uvt_projective_bundle_gauge_invariance_report.py
tests/test_star_uvt_projective_bundle_gauge_gradient_report.py
```

New mutation coverage rejects inconsistent row errors, stale summaries, bad
order derivative certificates, nonpositive gradient norms, and inconsistent
finite-difference fields.

## Verification

Syntax:

```text
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_gradient_report.py \
  tests/test_star_uvt_projective_bundle_gauge_invariance_report.py \
  tests/test_star_uvt_projective_bundle_gauge_gradient_report.py
```

Focused tests:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_bundle_gauge_invariance_report.py \
  tests/test_star_uvt_projective_bundle_gauge_gradient_report.py -q

21 passed in 6.45s
```

Saved artifact CLIs:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json

verified outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json
```

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_gradient_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json

verified outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json
```

## Decision Implications

This strengthens the answer to "why charts if we have fibers/gauge math?"

The fiber/gauge math defines the invariant object:

```text
UVT trace = pi_* Gamma^* rho
```

Charts or gauge domains are the regions where a chosen coordinate expression of
that invariant object has certified support, order, memory, and derivative
behavior. The stricter verifier protects that distinction: a coordinate change
is allowed only when it carries the measure and preserves the order certificate
needed by visibility.
