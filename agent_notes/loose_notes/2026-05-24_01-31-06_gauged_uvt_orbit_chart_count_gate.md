# Gauged UVT Orbit Chart Count Gate

Date: 2026-05-24 01:31:06

## Expansion Pass 1: Orbit Chart-Count Gate

## Context

The handoff goal was to run the first synthetic orbit chart-count gate for the
Gauged UVT Trace Atlas work:

```text
Does split_projective_trace_windows produce chart counts that grow with trace
complexity rather than linearly with frame density, while denominator-boundary
cases are marked unresolved?
```

The relevant compiler helper is:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
```

No shader files were edited in this pass.

## Current Model

The synthetic orbit should be a real projective camera chart, not just an
arbitrary screen polynomial. I used a static point under yaw, parameterized by
the stereographic orbit coordinate:

```text
t = tan(theta / 2)
cos(theta) = (1 - t^2) / (1 + t^2)
sin(theta) = 2t / (1 + t^2)
```

After multiplying by the common denominator, the homogeneous trace is exactly
quadratic:

```text
h_u(t) = x + 2 z t - x t^2
h_v(t) = y + y t^2
h_z(t) = z - 2 x t - z t^2
u(t) = h_u(t) / h_z(t)
v(t) = h_v(t) / h_z(t)
```

This is a good Gate B probe because it is a genuine rational camera trace and
has a real projective denominator boundary.

## What Changed

Added an exact denominator-root certificate to `ProjectiveTraceFit`:

```text
denominator_has_root
```

and made `split_projective_trace_windows(...)` reject windows whose continuous
chart domain contains a root of:

```text
h_z(t) = z0 + z1 t + z2 t^2
```

Important backtrack: the old validity rule only looked at sampled denominators.
That can accept two windows on either side of a denominator root if the root
falls between adjacent frame samples. The splitter now checks the continuous
interval through the next sample boundary so the projective chart boundary is
not hidden by frame sampling.

Added:

```text
tests/test_star_uvt_projective_orbit_windows.py
```

The tests cover:

- frame-density scaling at fixed visible orbit span
- trace-complexity scaling at fixed frame density
- denominator root between samples marked unresolved
- fit-level denominator-root certificate even when all samples are valid

## Evidence

For the visible yaw-orbit span `[-75deg, 75deg]` with affine local charts and
`max_residual_uv = 0.015`, chart counts versus frame density were:

```text
F=16  -> 4 accepted windows
F=32  -> 4 accepted windows
F=64  -> 4 accepted windows
F=128 -> 4 accepted windows
F=256 -> 4 accepted windows
```

The chart count is already saturated at the coarsest tested frame density for
this smooth span. It does not grow with `F`.

At fixed `F=128`, chart counts versus orbit span were:

```text
span=15deg  -> 1 accepted window
span=30deg  -> 2 accepted windows
span=60deg  -> 3 accepted windows
span=90deg  -> 7 accepted windows
span=120deg -> 7 accepted windows
```

This supports the desired model: accepted chart count tracks trace complexity
more directly than requested frame count.

At fixed 90-degree span, tightening residual raises chart count:

```text
max_residual_uv=0.08  -> 2 accepted windows
max_residual_uv=0.015 -> 6 accepted windows
max_residual_uv=0.005 -> 8 accepted windows
```

The denominator-boundary test uses a linear denominator root at `t=0.3` between
two valid samples:

```text
times = [0.0, 0.6]
h_z(t) = t - 0.3
```

The splitter now returns an unresolved window with reason:

```text
denominator_boundary
```

## Tests

Focused projective trace gate:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py -q
```

Result:

```text
14 passed in 1.03s
```

## Decision Implications

Supported:

- Gate B can produce chart counts that scale with rational trace complexity
  rather than directly with frame density on a smooth projective orbit probe.
- Denominator roots must be treated as chart boundaries in continuous time, not
  only as invalid sampled frames.

Still unresolved:

- These windows only certify trace centers/depth denominators, not footprint
  support bounds.
- The splitter is still compiler-side CPU/Torch code and is not wired into
  binning or rendering.
- Visibility/order strata still need depth uncertainty and crossing tests.

## Next Gate

Add support-bound tests:

```text
For every accepted rational/projective chart window, sampled u/v/depth traces
stay inside the compiled tile-time support bound.
```

Then add denominator/depth sidecars for visibility before attempting renderer
integration.
