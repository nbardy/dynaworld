# UV orbit-parameterized split-vs-fallback measurement

## Context

The adaptive UV split report had a synthetic high-motion frame sweep that
proved child-grid compilation can reduce fallback. The next question was
whether the same report can be driven by the revolving-camera gauge coordinate,
not just a plain frame index.

## Fixture

Added `test_orbit_derived_uv_visibility_split_report_reduces_fallback` in
`tests/test_star_uvt_projective_orbit_windows.py`.

The fixture uses:

```text
q = tan(theta / 2)
```

and the orbit depth polynomial from `_orbit_trace_coeffs(...)`. A second trace
adds a moving depth offset over:

```text
q in {-0.5, 0, 0.5}
```

so the pairwise UV order root moves across an 8-pixel parent tile. This ties
the UV split-vs-fallback report to the orbit gauge variable used by the
projective chart tests.

## Result

At the parent grid:

```text
parent_uv_event_tile_samples = 3
parent_fallback_fraction = 1.0
```

The adaptive policy tries child tile sizes `(4, 2, 1)`, chooses child size `2`,
and gets:

```text
residual_uv_event_tile_samples = 0
fallback_fraction = 0.0
```

This is still synthetic. It does not claim real-scene coverage, but it moves
the measurement from a generic high-motion line sweep into the revolving-camera
coordinate system.

## Verification

Targeted orbit/adaptive checks:

```text
4 passed in 4.13s
```

Focused STAR UVT projective plus interval-gated trainer suite:

```text
159 passed in 24.31s
```

## Next

Use the same parent/output fallback report on real high-motion traces. If
divisor-grid refinement leaves high residual fallback or a large cell-count
increase, that is the evidence for adding an oblique/fiber halfspace cell.
