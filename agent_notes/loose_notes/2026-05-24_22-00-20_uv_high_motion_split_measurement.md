# UV high-motion split-vs-fallback measurement

## Context

The adaptive UV split policy could already choose a child grid and report
residual events, but it did not expose the parent fallback baseline. That made
it awkward to answer the next research question: did grid refinement actually
reduce fallback, or did it merely move ambiguity around?

## Change

`ProjectiveTraceCellUVVisibilitySpatialSplitReport` now includes:

```text
parent_cells
parent_fallback_cells
parent_fallback_fraction
fallback_cells
fallback_fraction
```

This makes each adaptive split decision a measured before/after comparison.

## Synthetic high-motion fixture

Added a two-sample UV-line sweep in
`tests/test_star_uvt_projective_visibility.py`.

The pairwise depth line crosses different parts of one 8-pixel parent tile at
the two samples. The parent atlas therefore has:

```text
parent_uv_event_tile_samples = 2
parent_fallback_cells = 1
parent_fallback_fraction = 1.0
```

The adaptive policy tries divisor child grids `(4, 2, 1)`. Child size `4` is
still too coarse for one sample, so the policy chooses size `2`, producing four
child cells with:

```text
residual_uv_event_tile_samples = 0
fallback_cells = 0
fallback_fraction = 0.0
```

This is still synthetic, but it is the first explicit fallback-reduction
measurement for moving UV order lines.

## Verification

Targeted policy checks:

```text
4 passed in 1.24s
```

Focused STAR UVT projective plus interval-gated trainer suite:

```text
157 passed in 23.47s
```

## Next

Run the same before/after report on orbit-derived and real high-motion traces.
If divisor-grid refinement leaves a high residual fallback fraction or explodes
cell count, that is the evidence threshold for an oblique/fiber halfspace cell
representation.
