# Projective Bundle Gauge Gradients

## Context

The previous bundle-gauge artifact proved value invariance:

```text
pi_* Gamma^* rho
```

is numerically unchanged when the same revolving-camera primitive is integrated
in ordinary depth versus log-depth, provided the fiber-measure Jacobian is
included. The active goal also asks for clean derivatives and backward sharing,
so the next missing theory-contract gate was derivative invariance.

## Work

I added:

```text
research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_gradient_report.py
tests/test_star_uvt_projective_bundle_gauge_gradient_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.md
```

The report uses the same orbit camera and differentiates a weighted
fiber-pushforward objective with respect to:

```text
mean
log_precision
log_amplitude
```

It compares gradients from:

```text
ordinary depth integral
log-depth integral with dz/dr = exp(r)
log-depth integral without the Jacobian
```

## Evidence

Saved artifact:

```text
value_rel_error with Jacobian: 5.54e-14
max_gradient_rel_error with Jacobian: 2.33e-12
min_bad_no_jacobian_gradient_rel_error: 0.592
finite_difference_mean_x_rel_error: 1.42e-10
```

Verification:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_bundle_gauge_invariance_report.py \
  tests/test_star_uvt_projective_bundle_gauge_gradient_report.py -q

15 passed in 12.01s

PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_bundle_gauge_gradient_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_bundle_gauge_gradient/summary.json
```

## Interpretation

This is still a CPU/Torch math-contract gate, not a Metal speed row. The value
is that it protects the adjoint meaning of a screen-fiber gauge:

```text
valid gauge transition = same trace value + same primitive gradient
```

The missing-Jacobian control being wrong by `>=0.592` means the verifier should
catch exactly the bug that would poison a training/backward path while leaving
the code superficially "reasonable." This is a small but real step toward the
clean-derivatives part of the all-night objective.
