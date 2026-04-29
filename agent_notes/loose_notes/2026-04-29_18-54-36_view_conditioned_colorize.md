# View-Conditioned Feature Colorizer

## Context

The feature-splatting path renders an `F`-channel feature buffer and decodes it to
RGB with `FeatureToColor`. We wanted an optional camera/ray-conditioning knob for
specular or view-dependent effects without changing the default feature-splatting
behavior.

## Change

- Added `colorize.view_condition` with modes:
  - `"none"`: decode only the rasterized feature channels.
  - `"camera_center_ray"`: append one normalized world camera-forward direction,
    broadcast to every pixel in that frame.
  - `"pixel_ray"`: append each pixel's normalized world ray direction at render
    resolution.
- Added `colorize.detach_view_condition` defaulting to `true`, so the color MLP
  does not introduce a new camera-pose gradient path unless explicitly requested.
- Kept `pre_norm` scoped to the learned feature channels; ray channels are
  concatenated after feature normalization.
- Updated the colorize init probe to call the real `forward_with_logits` path,
  so diagnostics now include pre-norm and optional view conditioning.
- Documented the config options in
  `src/train_configs/local_mac_unconditioned_tokens_features_F32.jsonc`.

## Validation

- `python3 -m py_compile src/train/colorize.py src/train/train_video_token_implicit_dynamic.py src/train/probe_colorize_init.py`
- Shape probes for `none`, `camera_center_ray`, and `pixel_ray` all produced
  `[N, 3, H, W]` output with expected input dims (`F` or `F+3`).
- A one-step offline smoke with `colorize.view_condition="pixel_ray"` completed.
- `env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_fast_mac_feature_background.py`
