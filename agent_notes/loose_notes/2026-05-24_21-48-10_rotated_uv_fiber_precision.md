# Rotated UV Fiber Precision

## Context

The user pushed back on treating revolving-camera/orbit failures as merely
"fit residual checks plus fallback." The stronger theory target is that a
camera-gauge/fiber-bundle formulation should carry richer local geometry before
it gives up. In implementation terms, the projective interval path already
learned anisotropic `q_uu/q_vv`, but the source-view q-UVT tube model still
defaulted to an axis-aligned screen precision block.

Pinned memory for future agents:

```text
goal       fast 2D rasters across time from 4D spacetime primitives
meta-goal  share projection/support/binning/visibility/backward over time
key math   UVT trace = pi_* Gamma^* world_primitive
theory     STAR UVT is one local gauge expression of a camera-ray bundle atlas
```

## Current Model

A local camera gauge should be allowed to represent a rotated UV ellipse:

```text
Q_uv = [[q_uu, q_uv],
        [q_uv, q_vv]]
```

not only an axis-aligned diagonal. For revolving cameras, the projected
spacetime primitive can shear/rotate across the sensor fiber even when its
world-space primitive is simple. This does not remove the need for chart
splitting, but it raises the validity radius before a chart has to split.

## Implementation

`FeatureScreenTimeTubeModel` now owns `raw_spatial_correlation`, with

```text
rho  = rho_max * tanh(raw_spatial_correlation)
q_uv = rho * sqrt(q_uu q_vv)
```

and `rho_max < 1`, so the UV precision block is SPD by construction:

```text
det(Q_uv) = q_uu q_vv (1 - rho^2) > 0.
```

The UVT coupling terms are updated consistently:

```text
q_ut = -(q_uu v_u + q_uv v_v)
q_vt = -(q_uv v_u + q_vv v_v)
q_tt = q_t + q_uu v_u^2 + 2 q_uv v_u v_v + q_vv v_v^2
```

This preserves the recovered center velocity from the Schur block, so changing
the footprint gauge does not secretly change tube motion.

The projective interval trainer lock now also zeros/masks
`raw_spatial_correlation`; the anisotropic opt-in leaves it trainable.
Visibility support birth/split resets the raw cross term for reallocated tubes.

## Verification

Focused gate:

```text
tests/test_star_uvt_projective_uvt_producer.py::test_feature_tube_model_has_spd_trainable_uv_cross_precision
tests/test_star_uvt_projective_uvt_producer.py::test_projective_interval_trainer_bridge_backprops_with_locked_spatial_precision_if_available
tests/test_star_uvt_projective_uvt_producer.py::test_projective_interval_trainer_bridge_can_train_anisotropic_spatial_precision_if_available

3 passed in 4.90s
```

Syntax/import gate:

```text
py_compile passed for model, trainer, visibility support, interval backend,
native Python wrappers, and projective tests.
```

Broad gate:

```text
projective/interval suite: 154 passed in 23.49s
```

## Decision Implications

This makes the "rich math/gauge first" answer more real: a local STAR UVT chart
now has a full SPD screen-fiber metric under opt-in, while the default path
keeps legacy isotropic behavior. The next meaningful orbit/revolving-camera
gate should measure when rotated footprints reduce support drift, tile splits,
and fallback rate compared with diagonal anisotropy.
