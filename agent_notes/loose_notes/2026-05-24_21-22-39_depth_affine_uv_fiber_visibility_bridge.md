# Depth Affine UV Fiber Visibility Bridge

Date: 2026-05-24 21:22:39

## Context

The heartbeat continuation picked up after the `depth_affine_uv` contract had
been added to `ProjectiveTraceCellTraceAtlas`. Two explorer reports had
identified the remaining gap:

- the visibility certificate path needed to reason about depth over the whole
  tile, not only at trace centers;
- q-UVT lowering was still rejecting nonzero spatial `depth_beta[:,0:2]`,
  even though that is exactly the local screen-fiber depth plane we wanted.

The working goal remains the Gauged UVT Trace Atlas: compile 4D spacetime
primitives through a known camera program into reusable sensor-time traces, so
projection/support/binning/visibility/backward metadata can be shared across
time.

## Current Model

`depth_affine_uv` is now the first concrete "UVT screen fiber depth section":

```text
depth_affine_uv[N,6] = [zu0,zu1,zu2,zv0,zv1,zv2]

z(u,v,t) = z_c(t)
         + z_u(t) (u - u_c(t))
         + z_v(t) (v - v_c(t))
```

For compatible q-UVT tubes with affine world/screen depth:

```text
z(u,v,t) = depth0
         + beta_u (u - ma_u)
         + beta_v (v - ma_v)
         + beta_t (t - ma_t)
```

the producer now lowers to:

```text
center_depth_slope = beta_t + beta_u velocity_u + beta_v velocity_v
depth_affine_uv    = [beta_u,0,0,beta_v,0,0]
```

This keeps the moving trace center depth correct while preserving off-center
pixel depth for visibility.

## Code Changes

- `uvt_tubes_to_projective_trace_cell_atlas(...)` now has
  `allow_depth_affine_uv`. Default behavior remains conservative: spatial
  depth slopes are rejected unless the caller opts in.
- `src/train/star_uvt_projective_interval_backend.py` has
  `feature_uvt.projective_interval.allow_depth_affine_uv`, and live atlas
  updates recompute/preserve `depth_affine_uv` when the reference atlas has it.
- `projective_trace_cell_atlas_visibility_report(...)` and
  `mark_projective_trace_cell_visibility_fallbacks(...)` accept image/tile
  dimensions and use tile pixel-corner depth ranges when `depth_affine_uv` is
  present.
- The CPU/Torch reference fallback renderer sorts fallback tile/sample regions
  by live per-pixel depth when depth slopes are nonzero.
- The interval Metal op ABI includes `depth_affine_uv`; the hot interval
  selection sort evaluates `projective_cell_depth_at_pixel(...)`. The native
  extension had to be rebuilt so Python and C++ agreed on the new schema.

## Tests

Narrow targeted gate:

```text
7 passed in 9.54s
```

Covered:

- tile-range visibility detects an order flip hidden by center-depth sorting;
- fallback reference render sorts per-pixel depth and flips red/blue order
  across a tile;
- q-UVT producer rejects spatial depth by default;
- q-UVT producer opt-in lowers spatial depth into `depth_affine_uv`;
- live atlas updates preserve/recompute the depth plane;
- existing depth-at-UV helper and support-rebin preservation still pass.

Broad focused STAR UVT projective suite after rebuilding the native extension:

```text
146 passed in 34.01s
```

The first broad run failed with:

```text
render_projective_trace_cell_interval_tiles expected 13 args but received 14
```

That was stale `_C.cpython-311-darwin.so`, not a logic failure. Source
registration already had the new `depth_affine_uv` schema. Rebuilding
`third_party/fast-mac-gsplat/variants/star_uvt_v0` repaired it.

## Assumptions

- `depth_affine_uv` is compiled metadata / certificate structure for now.
  Native VJP does not return gradients for the depth-plane slopes.
- The current tile-range depth bound samples pixel-center tile corners. This is
  exact for affine-in-UV slopes over rectangular tiles and fixed time sample.
- For richer depth sections, the same certificate should become an event-root
  or conservative optimization problem rather than a corner-only check.

## Implications

This makes the "fiber bundle" answer less hand-wavy: the screen fiber is now
visible in code as a local depth section over `(u,v,t)`, and both compiler
certificates and interval Metal sorting can consume it. It does not eliminate
event domains; it makes each domain richer.

Next gates:

1. Add event roots for UV-varying depth order where possible, not just interval
   overlap fallback.
2. Decide whether depth-plane slopes should ever be differentiable parameters
   or remain compiler metadata derived from world primitives.
3. Extend the same depth-section contract to WorldFoam/instance-cell traces.
