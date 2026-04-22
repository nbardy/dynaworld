# Separated implicit camera variant

Session on 2026-04-22. User clarified the desired implicit-camera
architecture:

- do not feed Pluecker rays / GT camera into the input tokens;
- predict camera from an early video/image encoder branch;
- keep camera/path tokens out of the world-token self-attention stack;
- use predicted camera only at render/reconstruction time.

## Implementation

Added `src/train/gs_models/dynamic_token_gs_separated_implicit_camera.py`.

The new `DynamicTokenGSSeparatedImplicitCamera` reuses existing blocks and
heads:

- `ConvImageEncoder` and `TokenAttentionBlock` from `gs_models/blocks.py`
- `CanonicalGaussianParameterHeads` from the current image implicit-camera
  model
- `GlobalCameraHead`, `PathCameraHead`, and
  `compose_camera_with_se3_delta` from `gs_models/implicit_camera.py`

Flow:

```text
image -> ConvImageEncoder -> feature map
pooled early feature -> camera/path MLP heads -> predicted cameras
learned splat tokens + time -> TokenAttentionBlock(context=image features)
refined splat tokens -> Gaussian heads
render Gaussians with predicted cameras
```

The current joint implicit-camera model remains available as
`model.variant = "joint_attention"`. The new separated branch is selected with
`model.variant = "separated_camera"` in
`src/train_configs/local_mac_overfit_image_implicit_camera_separated.jsonc`.

## Verification

- `uv run python -m py_compile` on the new model, trainer, and model package
  init.
- Tiny forward smoke for both `DynamicTokenGSImplicitCamera` and
  `DynamicTokenGSSeparatedImplicitCamera`: 3 frames, 4 splat tokens,
  2 Gaussians/token; verified decoded Gaussian shape and predicted cameras.
- Config-selection smoke confirmed:
  - existing `local_mac_overfit_image_implicit_camera.jsonc` -> joint model
  - new `local_mac_overfit_image_implicit_camera_separated.jsonc` ->
    separated model

No full training run was launched.

## Worktree note

There were other unrelated dirty files in the worktree, including ongoing
docs and `src/train/train_video_token_implicit_dynamic.py`. They were left
alone. `uv run` briefly added `opencv-python` to `pyproject.toml` /
`uv.lock`; that accidental dependency diff was removed.
