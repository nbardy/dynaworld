# Spatial Precision VJP

## Goal Memory

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Context

The projective interval Metal path could render anisotropic
`spatial_precision_uv = (q_uu,q_uv,q_vv)`, but treated it as fixed compiled
metadata. That was fine for a pure playback compiler, but too weak for the
thread goal's clean-derivative condition. The UV footprint metric affects alpha
smoothly, so it should have a native VJP.

## Math

For a pixel offset `d=(du,dv)`:

```text
r^2 = q_uu du^2 + 2 q_uv du dv + q_vv dv^2
alpha = opacity * time_scale * exp(-0.5 r^2)
```

With fixed visibility/order:

```text
d alpha / d q_uu = -0.5 alpha du^2
d alpha / d q_uv = -alpha du dv
d alpha / d q_vv = -0.5 alpha dv^2
```

This is cleanly differentiable. It is unlike `depth_affine_uv`, which currently
controls discrete order selection and remains certificate metadata rather than
a smooth trainable target.

## Code

Changed:

```text
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/bindings.cpp
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/shared/common.h
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm
third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal
third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py
third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py
```

The native direct backward now returns:

```text
grad_coeffs
grad_opacity
grad_opacity_time_coeffs
grad_spatial_precision_uv
grad_color
```

`ProjectiveTraceAtlasGrad` exposes `grad_spatial_precision_uv`, and the
trainer-harness autograd wrapper passes `spatial_precision_uv` as a real input
when present. If an atlas has no precision metadata, wrappers still synthesize
the isotropic scalar precision and do not expose a gradient for it.

## Verification

Targeted tests:

```text
tests/test_star_uvt_projective_correctness.py::test_projective_cell_trace_interval_atlas_backward_uses_spatial_precision_uv_if_available
tests/test_star_uvt_projective_uvt_producer.py::test_uvt_tube_anisotropic_interval_autograd_backprops_to_spatial_precision_if_available

2 passed in 4.17s
```

Compile check:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/star_uvt_projective_interval_backend.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py
```

Broad projective/interval suite:

```text
152 passed in 16.41s
```

## Boundary

This makes the anisotropic screen-fiber metric trainable through the interval
Metal renderer. It does not make visibility order differentiable, and it does
not add gradients into `depth_affine_uv`; those remain chart/certificate
metadata unless a later design introduces a soft-order or explicit order-event
loss.
