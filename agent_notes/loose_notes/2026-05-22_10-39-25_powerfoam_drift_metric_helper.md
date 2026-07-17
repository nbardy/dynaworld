# PowerFoam Drift Metric Helper

## Context

Continuation of the modular trainer cleanup goal. The target was a live
repeated diagnostic payload, not a training-loop rewrite: PowerFoam Metal and
both Dynamic PowerFoam Metal model variants each rebuilt the same
center/radius/density/feature/normal/texel-site drift metrics before adding
their model-specific temporal, camera, token, quaternion, or texel-SV metrics.

## Change

- Added `src/train/powerfoam_diagnostics.py`.
- Added `powerfoam_parameter_delta_metrics(...)` for the common scalar payload:
  center mean/p95/max drift, xy/z drift, radius/density/feature drift, optional
  normal drift, optional texel-site drift, and optional cell count.
- Routed `MetalPowerFoamVideo.parameter_drift_metrics(...)`,
  `DynamicMetalPowerFoamVideo.parameter_drift_metrics(...)`, and
  `TokenDynamicPowerFoamFeatures.parameter_drift_metrics(...)` through that
  helper.
- Left temporal motion metrics, camera compact metrics, token RMS/static-dynamic
  counts, quaternion drift, tangent drift, texel height, and texel-SV metrics in
  the owning classes because those are not the shared contract.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/powerfoam_diagnostics.py src/train/train_powerfoam_metal.py src/train/train_dynamic_powerfoam_metal.py`
- `rtk uv run --with pytest --with torch python -m pytest tests/test_dynamic_powerfoam_metal.py::test_dynamic_powerfoam_implicit_camera_rays_are_per_frame_and_trainable tests/test_powerfoam_direct.py::test_powerfoam_metal_resample_uses_ema_and_preserves_optimizer_state tests/test_powerfoam_direct.py::test_powerfoam_metal_resample_prunes_invalid_cells -q`

Focused pytest result: `3 passed in 6.35s`.

## Handoff

This is a small diagnostic-helper extraction. It does not claim PowerFoam or
Dynamic PowerFoam convergence. The next useful cleanup should continue to
prefer this size of boundary: a repeated payload or setup contract with
trainer-specific math left where it belongs.
