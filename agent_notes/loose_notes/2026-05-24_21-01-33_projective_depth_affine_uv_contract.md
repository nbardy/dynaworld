# Projective Depth-Affine UV Contract

## Context

The Gauged UVT Trace Atlas thread needed a more formal answer to the user's
"screen fiber" idea: local projective/gauge math should carry richer
conditional depth than a scalar trace center, but without overclaiming that
all visibility is solved in the hot Metal path.

Previous work already added anisotropic screen precision metadata
(`spatial_precision_uv`) and aggregate support-tail alpha certificates. This
session added the depth-side companion.

## Implementation

`ProjectiveTraceCellTraceAtlas` now has optional:

```text
depth_affine_uv: Tensor[N,6]
```

with row layout:

```text
[zu0, zu1, zu2, zv0, zv1, zv2]
```

The model is:

```text
z(u,v,t) = z_c(t)
         + z_u(t) * (u - u_c(t))
         + z_v(t) * (v - v_c(t))

z_u(t) = zu0 + zu1 t + zu2 t^2
z_v(t) = zv0 + zv1 t + zv2 t^2
```

This is a local screen-fiber depth section over the projective UVT chart.

Files touched:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/__init__.py
tests/test_star_uvt_projective_correctness.py
tests/test_star_uvt_trainer_interval_gated.py
```

## Contract

The metadata validator requires:

```text
shape == [N,6]
dtype == float32
device == coeffs.device
contiguous
```

The helper:

```text
eval_projective_trace_cell_depth_at_uv_torch(...)
```

evaluates the pixel-varying conditional depth. If `depth_affine_uv` is absent,
it returns center depth, preserving the previous scalar-depth behavior.

The field is preserved through:

```text
support rebinning
trainer refresh
quadrature lowering
rolling/interval transformations where atlas metadata is carried
detached CPU conversion
```

## Important Boundary

This is not yet production pixel-varying visibility. The interval Metal
renderer still sorts/composites using the existing scalar/cell depth contract.
The new field is a compiler/certificate surface for the next visibility path.

Claim only:

```text
screen-fiber conditional depth is represented, validated, preserved, and
Torch-evaluable.
```

Do not claim yet:

```text
Metal interval sorting consumes depth_affine_uv.
```

## Verification

Focused depth/metadata tests:

```text
5 passed in 1.37s
```

After rebuilding `star_uvt_v0` so the native op schema matched the current
source, the broad focused projective plus interval-gated trainer slice passed:

```text
134 passed in 16.99s
```

The initial broad run failed because Python passed the newer
`spatial_precision_uv` argument into an older loaded `.so` whose
`render_projective_trace_cell_interval_tiles` schema still expected the old
12-argument form. Rebuilding fixed the ABI mismatch.

## Next

1. Use `depth_affine_uv` in visibility certificates by bounding
   `z_i(u,v,t)-z_j(u,v,t)` over tile-time cells.
2. Add a Metal-side path that consumes pixel-varying depth during interval
   sorting/fallback decisions.
3. Extend producers so real projective/gauged primitives can populate
   nonzero depth slopes, not only carry the metadata.
4. Keep anisotropic footprint precision and depth-plane visibility separate:
   support errors and order errors have different certificates.

Two read-only explorers were spawned for the next continuation:

```text
Explorer A / Hume / 019e5a4c-930c-7493-911f-ff6b909a332e
    Map the visibility/Metal hot path for consuming depth_affine_uv.

Explorer B / Laplace / 019e5a4c-b7b2-7a00-9bec-e2aca79e70fa
    Map producer/lowering paths that should generate nonzero depth_affine_uv.
```

They had not returned before this note was written.
