# STAR dense prototype runtime device boundary

## Context

`star_uvt_runtime.py` already owns STAR UVT checkout/path setup plus stricter
requested-device resolution and synchronization. The dense feature-tube
prototype still had local `_resolve_device(...)` and `_sync_device(...)`
wrappers that called `train_devices` directly.

## Change

- `research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py`
  now imports `_resolve_device` / `_sync_device` from `star_uvt_runtime`.
- The visibility bridge and birth/split prototypes continue to import those
  names from the dense prototype for compatibility, but the actual device
  policy now flows through the STAR runtime boundary.

## Validation

- `py_compile` covered the dense prototype and dependent support prototypes.

## Follow-up

The remaining STAR feature-tube scripts that import `resolve_device` directly
from `star_uvt_runtime` are already on the intended boundary. Do not introduce
another trainer-local or prototype-local device policy.
