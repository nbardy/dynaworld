# Depth-Affine Interval Metal Ordering

## Goal Memory

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Context

`ProjectiveTraceCellTraceAtlas.depth_affine_uv` already existed as the
screen-fiber conditional-depth section:

```text
z(u,v,t) = z_c(t)
         + z_u(t) (u - u_c(t))
         + z_v(t) (v - v_c(t))
```

CPU helpers, visibility certificates, q-UVT lowering opt-in, live atlas
updates, and quadrature lowering preserved it. The remaining gap was the hot
interval Metal path: it still selected front-to-back order from center depth.
That was mathematically wrong for revolving/projective gauges when conditional
depth varies across a tile.

## Implementation

The interval Metal ABI now carries:

```text
depth_affine_uv: Tensor[N,6] = [zu0,zu1,zu2,zv0,zv1,zv2]
```

through:

```text
render_projective_trace_cell_interval_tiles
render_projective_trace_cell_interval_rows
direct_projective_trace_cell_interval_backward
```

Python wrappers pass a dense zero tensor when the atlas has no depth plane, so
legacy scalar-depth atlases keep the same behavior.

The Metal order selector now computes:

```text
projective_cell_depth_at_pixel(...)
```

inside `select_projective_cell_order_id_interval(...)`, using the current pixel
center. Forward, row-gather forward, and direct backward all use the same
ordering. Depth-plane slopes are fixed compiled metadata; no gradients are
accumulated into `depth_affine_uv`.

## Test

Added a crossing-depth Metal test where two traces share the same center depth
but have opposite `z_u` slopes. The left pixel sees red in front; the right
pixel sees blue in front. This would fail if Metal sorted only by center depth.

Targeted:

```text
3 passed in 3.18s
```

Broad projective/interval suite:

```text
149 passed in 19.91s
```

Schema check confirms the custom ops include `depth_affine_uv` in forward and
backward.

## Boundary

This closes the immediate "screen fiber depth plane reaches Metal visibility"
gap. It does not yet make depth-plane slopes differentiable model parameters,
and it does not eliminate the need for visibility fallback where a tile has
important unresolved order changes. It does make the rich gauge math operational
in the hot path instead of falling back to center-depth ordering.
