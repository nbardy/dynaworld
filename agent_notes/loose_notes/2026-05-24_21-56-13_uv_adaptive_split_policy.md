# UV adaptive split-vs-fallback policy

## Context

The previous UV spatial split could retile a parent projective cell atlas onto
a manually chosen child tile grid. That proved the split side of the
UV-depth-line decision, but it was not yet a compiler policy: callers still had
to guess the child tile size and then separately inspect residual events.

The current packed atlas/render path assumes one global tile size per atlas.
That matters: we cannot yet mix a large parent tile for stable regions and a
small child tile only around the UV zero line unless we add a richer cell
coordinate schema or an oblique/fiber halfspace mask.

## Change

Added:

```text
adapt_projective_trace_cell_atlas_uv_visibility_events(...)
ProjectiveTraceCellUVVisibilitySpatialSplitReport
```

The policy:

1. Compute parent UV visibility events.
2. If none exist, keep the parent tile size and return the fallback-marked
   parent atlas.
3. If events exist, try divisor child tile sizes from coarsest to finest.
4. For each candidate, retile, recompute child depth intervals/order, run the
   UV event report again, and run fallback marking.
5. Accept the first candidate whose residual UV event-tile count is within the
   requested budget.
6. If no candidate clears the budget, return the best candidate with
   `accepted=False` and explicit `visibility_uv_depth_line` fallback on
   unresolved child cells.

The report stores input/output tile size, candidate sizes, parent/residual UV
event counts, output cell count, fallback cells, and fallback fraction.

## Tests

Added two focused invariants:

- A parent tile with a line that child size `2` resolves chooses output tile
  size `2` rather than over-splitting to `1`.
- A parent tile whose zero line still crosses a size-`2` child returns
  `accepted=False` and keeps one `visibility_uv_depth_line` fallback child.

Verification:

```text
targeted visibility checks: 4 passed in 3.06s
broad focused STAR UVT projective suite: 156 passed in 16.69s
```

## Decision implication

The UV visibility compiler now has an inspectable split-vs-fallback boundary.
The next research/prototype step is not "add fallback"; it is to measure on
orbit/high-motion scenes how often divisor child grids clear events, and where
grid refinement explodes enough to justify an oblique/fiber halfspace cell.
