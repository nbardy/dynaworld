# UV event-driven fallback marking

## Context

The prior step made UV visibility events explicit for projective STAR UVT
cells. For a trace pair inside a tile/sample, the compiler can now form

```text
Delta z(u, v, t_k) = a_u u + a_v v + a_0
```

and report an event when the affine range over the tile pixel-center rectangle
straddles zero. That turned "depth order may flip somewhere in this tile" into
a named geometric certificate instead of an inferred scalar-depth ambiguity.

The missing decision hook was fallback marking. Without it, the certificate
existed as diagnostics only, so a broad cell could still be accepted unless the
older interval overlap path happened to flag it.

## Change

`mark_projective_trace_cell_visibility_fallbacks(...)` now accumulates fallback
reasons per cell instead of a single anonymous ambiguous-cell set.

When an atlas has `depth_affine_uv` and tile depth domains, the fallback marker
calls `projective_trace_cell_uv_visibility_event_report(...)`. Every cell that
contains a UV zero-line event is marked with
`visibility_uv_depth_line`. If the same cell also has the older scalar/tile
depth overlap ambiguity, both reasons are preserved.

The important distinction:

- `visibility_ambiguous_depth`: interval/range order could not certify a stable
  order.
- `visibility_uv_depth_line`: the per-pixel affine depth plane has an actual
  zero line crossing the accepted tile.

This is still not spatial splitting. It is the correct fallback side of the
split-versus-fallback decision. The next gate is to add an oblique or
sub-tile split representation for cheap UV zero lines, then keep this fallback
reason for lines that are too expensive or chaotic to encode.

## Verification

Targeted visibility/projective checks:

```text
4 passed in 2.41s
```

Broad focused STAR UVT projective plus interval-gated trainer suite:

```text
151 passed in 10.09s
```

## Next

Add the spatial-split half of the UV visibility decision:

1. If the zero line is cheap to encode, split or stratify the tile so each side
   has stable order.
2. If the zero line is not cheap, keep `visibility_uv_depth_line` fallback.
3. Measure how much of orbit/revolving-camera stress moves from fallback into
   compiled stable-order cells.
