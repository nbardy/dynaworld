# Renderer Invalid-Depth Guard

Implemented the smallest stability fix for the 128px/4fps prebaked-camera NaN
without changing the camera model or adding a new loss.

## What Changed

- `src/train/renderers/common.py`
  - Added `MIN_RENDER_DEPTH`.
  - Near/behind Gaussians now get opacity zeroed and project through harmless
    finite placeholders before Jacobian/covariance math.
  - This avoids the old `z_safe=1e-4` explosion for behind-camera splats.

- `src/train/renderers/dense.py`
  - Clamps Gaussian exponent `power` to `<= 0`, which is the physical range for
    a positive semidefinite Gaussian exponent.
  - Added optional `return_aux=True` to return detached `alpha_max` and
    `weight_sum` summaries from tensors the dense renderer already computed.

- `src/train/rendering.py` and `src/train/dynamicTokenGS.py`
  - Threaded optional dense render aux through metric collection only.
  - Training default behavior remains normal tensor rendering unless metrics
    request aux.

- `src/train/debug_metrics.py`
  - Added detached render-aux diagnostics:
    - alpha-support Gaussian counts
    - contributing Gaussian counts
    - no-support/no-contribution slot fractions
    - alpha/weight nonfinite counts

- `src/train/sequence_data.py`
  - Made `cv2` lazy so camera-JSON/all-frame training does not require OpenCV
    at import time.

## What This Does Not Fix

The 128px late-frame geometry is still bad: many splats are near/behind camera
for the raw DUSt3R scale. The patch prevents NaNs so we can train/debug, but
diagnostics still expose the camera/scene contract violation.

## Checks

- `uv run python -m py_compile ...` passed for touched train/render files.
- Synthetic render/backward with one safe splat, one behind splat, and one near
  splat is finite.
- 32px default one-step smoke passed with finite render/loss.
- 128px random one-step smoke passed with finite render/loss.
- Forced old bad 128px frame window `[37, 38, 39, 40]` passed render/loss/backward
  finite.

Forced bad-window diagnostics still show bad geometry:

```text
CameraZMin=-1.01
CameraZMax=0.7077
FrontGaussiansMin=30
NearOrBehindGaussiansMean=398
RenderNonfiniteCount=0
PowerGt80Count=0
AlphaPreNonfiniteCount=0
NoContributionSlotFraction=0.7363
```

This is the desired interim state: stable numerics, honest diagnostics.
