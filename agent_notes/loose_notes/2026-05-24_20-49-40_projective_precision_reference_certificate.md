# Projective Precision Reference And Certificate Wiring

## Context

Goal memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The prior pass added `ProjectiveTraceCellTraceAtlas.spatial_precision_uv` and
proved the anisotropic rectangle tail-bound math in a standalone verifier. The
remaining risk was that the code would carry rich gauge/projective precision
metadata while the actual CPU oracle and refresh certificate kept behaving like
scalar `sigma_px`.

## Change

The CPU/reference trace evaluators now consume `spatial_precision_uv` when it
is present:

```text
alpha = opacity * exp(-0.5 * [du dv]^T P [du dv])
P = [[q_uu, q_uv],
     [q_uv, q_vv]]
```

This covers:

- `render_projective_trace_cell_atlas_reference`
- `render_projective_trace_cell_atlas_quadrature_reference`

The stale-support tail-alpha certificate in
`refresh_projective_cell_interval_atlas_if_stale` also uses the same precision.
For each omitted tile rectangle, it minimizes the Mahalanobis quadratic over
interior, stationary edge points, and corners, then sums omitted alpha per
tile/sample. This is the code version of the verifier math.

At this point in the session, production projective interval Metal still used
scalar `sigma_px`; a follow-up note records the later Metal bridge that changed
that for hand-built projective cell atlases. q-UVT lowering still protects its
compatibility path by rejecting non-isotropic spatial precision.

## Verification

Syntax:

```text
.venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_trainer_interval_gated.py
```

Focused precision tests:

```text
tests/test_star_uvt_projective_correctness.py::test_projective_cell_atlas_reference_uses_spatial_precision_uv
tests/test_star_uvt_projective_correctness.py::test_projective_cell_quadrature_reference_uses_spatial_precision_uv
tests/test_star_uvt_trainer_interval_gated.py::test_projective_interval_support_tail_alpha_certificate_uses_spatial_precision_uv

3 passed in 1.91s
```

Broad projective/interval suite:

```text
133 passed in 18.94s
```

Verifier reruns:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_precision_rerun/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_precision_rerun/summary.md
```

The scalar image-error verifier still rejects overlapping omitted tails
(`0.01309515 > 0.001`). The anisotropic verifier still passes diagonal,
rotated, and two-trace same-tile cases while rejecting omitted core support.

## Interpretation

This kept the theory honest at the CPU/certificate layer: the gauge/projection
math was no longer only stored in metadata; the CPU oracle and support reuse
certificate evaluated the local quadratic footprint. The immediate remaining
bridge was Metal footprint evaluation/backward for per-trace precision; that
bridge is recorded in `2026-05-24_21-02-21_projective_precision_metal_bridge.md`.
