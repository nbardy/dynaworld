# UV Visibility Event Certificate

Date: 2026-05-24 21:35:43

## Context

The previous pass wired `depth_affine_uv` through q-UVT lowering, tile-range
visibility checks, and interval Metal sorting. The next gap was conceptual:
when UV-varying depth creates an order boundary inside one tile, the compiler
could mark ambiguous/fallback, but there was no first-class event saying what
geometry caused that ambiguity.

## Current Model

At a fixed sensor-time sample, each trace with `depth_affine_uv` induces:

```text
z_i(u,v,t_k) = z_ci(t_k)
             + z_ui(t_k) (u - u_ci(t_k))
             + z_vi(t_k) (v - v_ci(t_k))
```

For a pair `(i,j)`, the order boundary over a tile is an affine line:

```text
Delta z_ij(u,v,t_k) = z_i - z_j
                    = a_u u + a_v v + a_0
```

If the min/max of this line over the tile's pixel-center rectangle straddles
zero, the tile contains a UV visibility event. A time split will not remove
that event; the compiler must either split spatially, lower the tile size, use
a sub-tile representation, or mark fallback.

## Code Added

New dataclasses:

```text
ProjectiveTraceCellUVVisibilityEvent
ProjectiveTraceCellUVVisibilityEventReport
```

New helper:

```text
projective_trace_cell_uv_visibility_event_report(...)
```

It reports `cell_index`, tile id, sample index, trace pair, sample time, line
coefficients, and min/max depth delta over the tile.

The report is exported from `torch_gsplat_bridge_star_uvt.__init__`.

## Tests

Added focused tests in `tests/test_star_uvt_projective_visibility.py`:

- in-tile depth line crossing is reported with line `0.4 u - 0.9`;
- a stable depth plane over the same tile reports no events.

While running the broad suite, a stale native backward ABI surfaced. Python
expected the current 5-return interval backward op including
`grad_spatial_precision_uv`, but the loaded extension still registered a
4-return schema. A forced rebuild fixed the schema.

Also fixed two existing spatial-precision backward tests:

- the generic backward test no longer assumes optional `spatial_precision_uv`
  exists;
- the precision-specific test now creates the reference precision tensor before
  comparing `grad_spatial_precision_uv`.

Verification:

```text
targeted UV/depth tests: 4 passed in 12.01s
spatial precision backward pair: 2 passed in 3.70s
broad focused STAR UVT projective suite: 150 passed in 14.23s
```

## Implication

This is a small but real step from "fallback when a chart cannot cover order"
to "the compiler knows the UV order boundary." The next meaningful step is a
decision rule:

```text
if UV event line is cheap to represent:
    split spatially / sub-tile
else:
    fallback
```

The event certificate gives that future decision something mathematical to
consume.
