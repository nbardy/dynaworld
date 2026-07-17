# Projective Spatial Precision Metadata

## Context

Goal memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The anisotropic tail-bound verifier proved the certificate math for SPD
screen-space footprints, but the compiled atlas still had nowhere to carry that
footprint shape. At this point in the session the production renderer was still
scalar `sigma_px`, so the safe next step was metadata plumbing, not pretending
anisotropic Metal was done.

## Change

`ProjectiveTraceCellTraceAtlas` now has optional:

```text
spatial_precision_uv: Tensor[N,3] | None
```

Rows store:

```text
(q_uu, q_uv, q_vv)
```

Validation requires float32, contiguous, same device as `coeffs`, shape
`[N,3]`, and positive definiteness:

```text
q_uu > 0
q_vv > 0
q_uu * q_vv - q_uv^2 > 0
```

Preserved through:

- support-event rebinning
- sampled support rebinning
- visibility event stratification
- visibility fallback marking
- quadrature lowering
- detached CPU reference conversion
- trainer-state materialization

q-UVT lowering and live-atlas updates populate the field from source `q_uvt`.
At this point in the session, the production interval Metal path still
rejected non-isotropic spatial precision before using its scalar renderer.

Follow-up in the same thread moved the metadata from inert contract to CPU
reference behavior. `render_projective_trace_cell_atlas_reference`,
`render_projective_trace_cell_atlas_quadrature_reference`, and
`refresh_projective_cell_interval_atlas_if_stale` now evaluate the footprint
with the stored SPD UV precision when present. A later note in the same session
records the production interval Metal forward/backward bridge for hand-built
projective cell atlases; q-UVT compatibility lowering remains isotropic.

## Verification

Compile:

```text
.venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py \
  src/train/star_uvt_projective_interval_backend.py \
  tests/test_star_uvt_trainer_interval_gated.py \
  tests/test_star_uvt_projective_uvt_producer.py
```

Focused precision tests:

```text
3 passed in 1.91s
```

Broad projective/interval suite:

```text
133 passed in 18.94s
```

Current-code verifier reruns:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_precision_rerun/summary.md
outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound_precision_rerun/summary.md
```

The scalar rerun includes an overlapping-tail aggregate rejection:

```text
aggregate omitted bound = 0.01309515
budget = 0.001
reuse = rejected
```

## Interpretation

This was stronger than metadata-only: CPU/reference rendering and support reuse
consumed the local screen footprint precision. The next bridge was to carry the
same per-trace precision into the production Metal interval kernels while
preserving the existing isotropic q-UVT parity path; that follow-up is now
recorded separately.
