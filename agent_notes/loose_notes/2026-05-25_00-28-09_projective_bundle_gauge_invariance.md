# Projective Bundle Gauge Invariance

## Context

The active goal is still the STAR UVT / WorldFoam camera-path compiler:
fast 2D rasters across time from 4D spacetime primitives, with projection,
support, binning, visibility, memory bandwidth, and backward work shared over
time. The theory notes already state the invariant:

```text
UVT trace = pi_* Gamma^* world_primitive
```

The missing piece was a small executable falsification test for the "UVT
screen fiber" / camera-ray bundle wording.

## Work

I added:

```text
research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py
tests/test_star_uvt_projective_bundle_gauge_invariance_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.md
```

The report integrates the same spacetime Gaussian through a revolving pinhole
camera in two gauges:

```text
z = camera-forward depth
r = log(z)
```

The expected invariant is:

```text
integral rho(Gamma(y,z)) dz
==
integral rho(Gamma(y,exp(r))) exp(r) dr
```

where `exp(r)` is the measure Jacobian.

## Evidence

Saved artifact:

```text
max_rel_error with Jacobian: 3.50e-13
min relative error without Jacobian: 0.600
monotone log-depth order preserved: true
orientation-reversing -log(depth) order flipped: true
```

The focused tests pass:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_bundle_gauge_invariance_report.py -q

7 passed in 5.52s
```

The saved artifact verifier passes:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_invariance_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_invariance/summary.json
```

## Interpretation

This is not a speed gate and not a Metal hot-path row. It is a theory-contract
gate. It makes the fiber-bundle claim testable:

- gauge changes are allowed when the fiber measure transforms correctly
- monotone depth gauges preserve local visibility order
- orientation-reversing depth coordinates are not valid visibility gauges and
  must become chart/gauge boundaries
- omitting the Jacobian is a large, visible bug

This supports the user's "rich math, not fallback-first" objection: the gauge
is carrying real structure before any chart split or fallback.
