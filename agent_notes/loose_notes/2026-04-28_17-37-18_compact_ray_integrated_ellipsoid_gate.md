# Compact Ray-Integrated Ellipsoid Gate

## Context

We evaluated a new representation proposal:

```text
compact ray-integrated ellipsoidal splat
```

with density:

```text
kappa(x) = beta * [1 - (x - mu)^T A (x - mu)]_+^k
```

The important claim is narrow:

```text
single-view RGB cannot remove radial depth / opacity-support gauge,
but analytic world-space ray integration can remove the projected-splat
screen-covariance nullspace.
```

This is not yet a full trainer result. It is a better primitive hypothesis and
a sharper local nullspace test.

## What Changed

Added an analytic helper in:

```text
research_experiments/gauge_fields/incidence.py
```

Function:

```text
compact_poly_ellipsoid_optical_depth
```

It computes finite-segment optical depth for:

```text
beta * [1 - (o + s d - mu)^T A (o + s d - mu)]_+^power
```

using the closed-form polynomial antiderivative after clipping the ray interval
to the compact ellipsoid support.

Added tests in:

```text
tests/test_gauge_incidence.py
```

The tests check:

```text
closed form matches numeric quadrature
radial gauge invariance holds exactly
projected Gaussian covariance-null perturbations are not null for compact
ray-integrated ellipsoids
```

## Verification

Passed:

```bash
uv run python -m py_compile research_experiments/gauge_fields/incidence.py tests/test_gauge_incidence.py
uv run --with pytest python -m pytest tests/test_gauge_incidence.py -q
```

Result:

```text
11 passed
```

## Read

This proposal is more precise than the prior `ray_gaussian_line_mass` path.

`ray_gaussian_line_mass` already moved opacity into a world-space line integral,
but Gaussian support is infinite and the current implementation still relies on
projected candidate bounds for speed. The compact polynomial ellipsoid gives:

```text
finite conic footprint
constant per-pixel analytic evaluation
direct local-rasterizer compatibility
a theorem-level reason the screen-covariance nullspace is gone
```

The remaining gauge is still real:

```text
mu -> lambda mu
A -> lambda^-2 A
beta -> lambda^-1 beta
```

Do not claim metric depth from a single RGB view. The point is only that this
primitive removes one avoidable degeneracy that projected splats keep.

## Next Integration

Do not immediately replace the gauge runner. The next safe integration is:

```text
render.incidence_mode = compact_poly_ellipsoid
```

under an existing world-covariance support mode:

```text
derived_support_metric
rank_adaptive_metric
transported_world_ball
```

For a covariance `Sigma`, use:

```text
A = inv(Sigma)
```

and calibrate `beta` so center-ray optical depth roughly matches the current
initialized alpha. Then run the same existing selector:

```text
3-camera DeepView train-2/test-1
same 2048 primitive count
heldout_eval_psnr / heldout_eval_l1
heldout coverage / X-map / witness diagnostics
wall clock
```

## Kill Criteria

Demote the compact ellipsoid path if:

```text
it only wins source-view metrics,
heldout PSNR gains come with the same broad-coverage pattern as ray_gaussian_line_peak,
X-map occupancy or xmap_shuffle red-team gets worse,
runtime is not close to projected_conic after a bounded footprint implementation,
or it still loses to derived_support_metric/projected_conic after beta calibration.
```

The first implementation target is not a new model family. It is an incidence
law that tests whether compact world support plus analytic line integration
beats projected-conic support without paying the current all-pairs Gaussian-line
runtime.
