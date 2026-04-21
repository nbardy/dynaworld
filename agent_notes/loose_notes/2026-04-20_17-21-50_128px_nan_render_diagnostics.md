# 128px NaN Render Diagnostics

The 128px/4fps prebaked-camera run can hit `NaN` at the first optimization
step depending on the sampled frame window. A read-only diagnostic loop found
the first failure with frame indices `[37, 38, 39, 40]`.

What failed:

- Decoded model tensors were finite at the failing step:
  - `xyz` roughly `[-0.81, 1.79]`
  - `scales` roughly `[0.023, 0.100]`
  - `opacities` roughly `[0.34, 0.64]`
- The dense renderer output became non-finite.
- Projection diagnostics showed late-frame 128px cameras put most Gaussians at
  or behind the camera:
  - frame `37`: `391 / 512` Gaussians near/behind
  - frame `40`: `496 / 512` Gaussians near/behind
- The renderer masks near/behind opacity, but still computes projection power
  first. Some projected covariance determinants become huge/negative from the
  near-plane math, producing enormous positive power values. Then
  `0 * exp(huge)` becomes `0 * inf = NaN`.

The 64px corrected bake did not show this initial pathology in the same metric
sweep. Its raw determinants stayed positive and power stayed non-positive.

Camera-pose comparison:

- 64px corrected camera path has relative translation z in about
  `[-0.28, 0.11]`.
- 128px corrected camera path has relative translation z up to about `2.56`,
  with frames `37-45` clustered around `z ~= 2.4-2.56`.
- The current `GaussianParameterHeads` constrains predicted z to `[0.5, 2.5]`
  in the first-camera coordinate system. The 128px camera solve can therefore
  move the camera nearly through/past the whole model output depth range before
  training has a chance to adapt.

Instrumentation added:

- `dense_render_diagnostics(...)` in `src/train/dynamicTokenGS.py`
- fail-fast checks for non-finite render/loss that log diagnostic metrics to
  W&B before raising
- `debug_render_metrics_every` in the 128px config, set to `25`

No stabilization fix was applied in this chunk. The likely real fix is to treat
DUSt3R translation scale as arbitrary and normalize camera translations into a
model-compatible coordinate range, then separately make the renderer robust to
culled/behind Gaussians so diagnostics fail cleanly instead of producing NaNs.
