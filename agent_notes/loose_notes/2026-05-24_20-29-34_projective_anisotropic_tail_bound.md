# Projective Anisotropic Tail-Bound Verifier

## Context

Goal memory:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

The previous tail-alpha certificate was isotropic: it bounded an omitted tile
using `opacity * exp(-0.5 * (distance / sigma_px)^2)`. That is enough for the
current scalar projective interval route, but real camera gauges and q-UVT
traces need anisotropic screen footprints.

## Change

Added:

```text
research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py
```

It implements the local anisotropic omitted-support certificate:

```text
alpha_tail(R)
<= opacity * exp(-0.5 * min_{x in R} (x - mu)^T P (x - mu))
```

where `R` is the omitted tile rectangle and `P` is a positive definite 2D
screen precision. For a rectangle, the convex quadratic minimum is found by
enumerating:

- interior center if it lies in the rectangle,
- stationary points on each vertical/horizontal edge,
- corners.

For multiple traces omitted in the same tile, the verifier sums the per-trace
tail bounds before comparing against the budget.

## Artifact

Run:

```text
.venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py \
  --tail-alpha-epsilon 0.001 \
  --out-dir outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound
```

Summary:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_anisotropic_tail_bound/summary.md
```

## Result

```text
diagonal_sigma_u1_v2_tail:
    bound 0.0002046116
    max error 0.0000242851
    reuse accepted

rotated_precision_tail:
    bound 0.0001845283
    max error 0.0000190703
    reuse accepted

two_trace_same_omitted_tile_sum:
    bound 0.0002287796
    max error 0.0000166098
    reuse accepted

anisotropic_core_loss_rejected:
    bound 0.5
    max error 0.4379515
    reuse rejected
```

## Verification

```text
.venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_anisotropic_tail_bound_verifier.py
```

The verifier exited successfully and wrote the artifact above. The scalar
tail-alpha image-error verifier also reran successfully to
`outputs/benchmarks/2026-05-24_star_uvt_projective_tail_alpha_image_error_rerun/summary.md`.

## Interpretation

This does not make the production Metal projective interval path anisotropic.
That route still takes scalar `sigma_px`, and q-UVT lowering still rejects
anisotropic spatial precision. The useful progress is sharper: we now have the
certificate math and an artifact for the next atlas metadata contract.

Next implementation bridge:

```text
ProjectiveTraceCellTraceAtlas should carry per-trace/per-cell screen precision P_uv.
Support reuse should use rectangle Mahalanobis bounds instead of scalar distance.
Metal/reference rendering then need anisotropic footprint evaluation.
```
