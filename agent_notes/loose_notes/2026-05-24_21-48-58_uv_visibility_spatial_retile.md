# UV visibility spatial retile

## Context

The UV visibility event certificate could already find an in-tile order
boundary:

```text
Delta z(u, v, t_k) = a_u u + a_v v + a_0
```

and the fallback marker could name that case with
`visibility_uv_depth_line`. That was correct but one-sided: it implemented the
fallback side of the split-versus-fallback decision, not the split side.

The existing cell schema does not yet carry oblique halfspace masks. Rather
than inventing an unrendered polygon representation, the first practical split
uses the schema we already have: a finer global tile grid.

## Change

Added:

```text
split_projective_trace_cell_atlas_uv_visibility_events(...)
```

The function takes a parent atlas plus `tile_size` and `child_tile_size`.
It retile-compiles every parent cell onto the child grid, recomputes each
child cell's depth interval over that child tile footprint, and stores the
child-local front-to-back order.

This makes a cheap UV depth-line split representable today:

```text
parent tile has a crossing line
    -> child tiles on each side get stable orders
    -> fallback marker sees no child UV event
```

If the zero line still crosses a child tile, the existing fallback marker keeps
the explicit `visibility_uv_depth_line` reason. So the decision stack is now:

```text
detect UV line
try grid-refinement spatial split
fallback only for unresolved child tiles
```

## Test invariant

The canonical fixture has two traces whose depth difference is:

```text
Delta z(u) = 0.4 u - 0.9
```

At parent tile size `4`, the line crosses the tile. At child tile size `2`,
the left child has stable order `(0,1)` and the right child has stable order
`(1,0)`. The retiled atlas has no UV event, no fallback, no visibility stale
report, and the reference render shows red/front on the left and blue/front on
the right.

## Verification

Targeted checks:

```text
4 passed in 1.80s
```

Broad focused STAR UVT projective plus interval-gated trainer suite:

```text
153 passed in 15.90s
```

## Open limits

This is still a grid-refinement split, not a fiber/halfspace cell atlas. It may
over-split when an oblique line would be cheap to encode analytically, and it
can still fall back if a child tile contains the zero line. The next real gate
is an adaptive policy: choose child tile sizes from line geometry and measure
split-vs-fallback fractions on orbit/high-motion scenes.
