# PowerFoam Metal Dynamic Geometry Proof Tests

Date: 2026-05-06

## Context

The subagent clarified that P0.3/P0.4 should not count feature/RGB time
conditioning as dynamic geometry. A real proof needs alpha/support changes from
time-conditioned geometry while color/features are frozen.

## What Was Already True

`DynamicMetalPowerFoamVideo` already has temporal coefficients for geometry on
the Metal path:

- `raw_xy_coeff`
- `raw_z_coeff`
- `raw_radii_coeff`
- `raw_densities_coeff`
- `raw_normals_coeff`
- `raw_tangents_coeff`
- `raw_texel_sites_coeff`

It also has `raw_features_coeff`, so the code can represent both geometry
motion and repainting. The missing artifact was proof that we can separate the
two.

## Added Tests

Updated `tests/test_dynamic_powerfoam_metal.py` with:

- zero-coeff static decode parity;
- geometry coefficients move centers/radii without changing features;
- feature coefficients change features without moving geometry;
- MPS render proof that geometry coefficient motion changes alpha.

The MPS alpha test keeps all coefficients zero, renders frames 0 and 1, checks
alpha parity, then changes only center/radius coefficients and asserts alpha
differs across time.

## Verification

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:third_party/dynamic-powerfoam-metal \
  uv run --with pytest python -m pytest -p no:cacheprovider \
  tests/test_dynamic_powerfoam_metal.py -q
```

Result:

```text
13 passed in 13.34s
```

## Remaining Gap

This proves Metal has minimal dynamic geometry mechanics and alpha causality. It
does not yet provide a nearest0040 multicam training row. The next P0.3 step is
a real dynamic-geometry config/run that logs geometry-time deltas,
alpha/support-time deltas, and a repaint-only control.

CUDA P0.4 is still appearance-only until the upstream fork has scene-side
center/radius/height coefficients and a Modal L40S comparison showing
alpha/support time changes.
