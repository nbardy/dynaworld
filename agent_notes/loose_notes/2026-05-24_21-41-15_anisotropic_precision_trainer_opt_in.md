# Anisotropic Precision Trainer Opt-In

## Goal Memory

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Context

After adding `grad_spatial_precision_uv` to the interval Metal VJP, the
source-view trainer still had one production-level brake:
`_lock_projective_interval_spatial_precision(...)` always forced
`raw_precision[:,0:2]` to `sigma_px^{-2}` and masked their gradients whenever
the projective interval route was enabled. That preserved the old isotropic
contract but prevented the new anisotropic metric derivative from affecting the
actual trainer.

## Change

`run_training(...)` now checks:

```text
projective_interval.allow_anisotropic_spatial_precision
```

Behavior:

```text
False/default:
    lock raw_precision[:,0:2] to sigma_px^{-2}
    register the spatial-gradient mask
    report projective_interval_spatial_precision_locked=True

True:
    skip the lock
    let q-UVT lowering carry spatial_precision_uv
    let the interval Metal VJP update raw_precision[:,0:2]
    report projective_interval_spatial_precision_locked=False
```

This makes the anisotropic screen-fiber footprint metric a real opt-in training
surface rather than only a hand-built atlas capability.

## Verification

Focused bridge tests:

```text
tests/test_star_uvt_projective_uvt_producer.py::test_projective_interval_trainer_bridge_backprops_with_locked_spatial_precision_if_available
tests/test_star_uvt_projective_uvt_producer.py::test_projective_interval_trainer_bridge_can_train_anisotropic_spatial_precision_if_available

2 passed in 8.76s
```

Compile check:

```text
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/star_uvt_feature_overfit_trainer.py \
  src/train/star_uvt_projective_interval_backend.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/projective_trace.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/trainer_harness/tile_metal_autograd.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py
```

Broad projective/interval suite:

```text
153 passed in 20.31s
```

## Boundary

This still leaves depth-affine visibility as compiled/certificate metadata:
`depth_affine_uv` affects dynamic sort order, but no smooth gradient is claimed
through that discrete ordering. The differentiable part now covered is the
anisotropic alpha footprint metric.
