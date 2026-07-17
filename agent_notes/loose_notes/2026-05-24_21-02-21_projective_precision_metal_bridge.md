# Projective Precision Metal Bridge

## Context

Goal memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The prior precision pass made `spatial_precision_uv` real for the CPU/reference
renderer and support certificate. The remaining "try it in Metal" step was to
thread the same fixed per-trace UV precision through the production interval
Metal kernels without disturbing the existing scalar/isotropic q-UVT route.

## Change

The interval cell Metal op ABI now accepts:

```text
spatial_precision_uv: Tensor[N,3] = (q_uu, q_uv, q_vv)
```

for:

- `render_projective_trace_cell_interval_tiles`
- `render_projective_trace_cell_interval_rows`
- `direct_projective_trace_cell_interval_backward`

Python wrappers synthesize `P = sigma_px^{-2} I` when the atlas has no
precision metadata, so existing scalar tests keep the old behavior. If the
atlas has `spatial_precision_uv`, the Metal kernels evaluate:

```text
radius2 = q_uu du^2 + 2 q_uv du dv + q_vv dv^2
alpha = opacity * opacity_time_scale * exp(-0.5 * radius2)
```

Backward treats precision as fixed compiled metadata and updates center
coefficients with:

```text
d alpha / d u_c = alpha * (q_uu du + q_uv dv)
d alpha / d v_c = alpha * (q_uv du + q_vv dv)
```

The q-UVT compatibility producer still locked/rejected non-isotropic spatial
precision at the time of this note. A later note,
`2026-05-24_21-14-57_anisotropic_quvt_projective_compiler.md`, adds the
explicit anisotropic q-UVT compiler/backend opt-in while keeping the learned
source-view trainer model lock isotropic by default.

## Verification

Rebuilt the extension:

```text
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Schema check:

```text
render_projective_trace_cell_interval_tiles(..., Tensor spatial_precision_uv, ...)
MPS available: True
interval forward op: True
interval backward op: True
```

Focused Metal precision parity:

```text
tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_forward_uses_spatial_precision_uv_if_available
tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_backward_uses_spatial_precision_uv_if_available

2 passed in 1.35s
```

Broad projective/interval suite:

```text
138 passed in 16.36s
```

Verifier reruns:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_metal_precision_rerun/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_metal_precision_rerun/summary.md
```

## Interpretation

This is the first actual Metal realization of the screen-fiber/gauge precision
math. It is still a fixed-metadata renderer: no gradient is accumulated into
`spatial_precision_uv`. A later q-UVT compiler opt-in can now populate
anisotropic precision, but the learned source-view trainer route still exposes
the isotropic training path by default. The core local formula lines up across
theory, reference render, support certificate, Metal forward, and Metal
backward.
