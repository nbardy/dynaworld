# Pose-To-Plucker Video-Token Architecture

## Context

The current video-token implicit-camera path had two swappable variants:

- `learned_time_orbit_path`
- `sinusoidal_time_path_mlp`

The user clarified that the camera latent should remain a D-dimensional token.
The desired direction was not to make the latent itself Plucker-valued, and not
to predict a 6D ray per pixel. The intended contract is:

1. decode a compact camera from latent camera/path tokens and explicit time
2. derive Plucker rays analytically from that camera
3. use those rays as downstream conditioning for the splat/world decoder

## Implemented Variant

Added `token_to_pose_to_plucker`.

The new path uses:

- `TimeConditionedOpticalAxisCameraHead`
  - input: path token + sinusoidal time embedding
  - output: bounded yaw/pitch/roll plus bounded local translation
  - construction: base orbit camera from the global camera token, then build a
    camera frame from explicit center, optical axis, and roll
- `PluckerRayTokenConditioner`
  - builds an analytic Plucker ray grid from the decoded camera
  - projects/downsamples the ray grid
  - cross-attends splat tokens to the ray context before Gaussian heads

This keeps the user's requested boundary: camera token is latent; geometry is
decoded once; full pixel ray fields are derived, not predicted.

## Files

- `src/train/gs_models/implicit_camera.py`
  - added `build_cameras_from_optical_axis`
  - added `TimeConditionedOpticalAxisCameraHead`
- `src/train/gs_models/dynamic_video_token_gs_implicit_camera.py`
  - added `PluckerRayTokenConditioner`
  - added `DynamicVideoTokenGSImplicitCameraPoseToPlucker`
- `src/train/gs_models/__init__.py`
  - exports the new variant
- `src/train/train_video_token_implicit_dynamic.py`
  - maps `model.variant = "token_to_pose_to_plucker"` to the new class
  - adds `ray_condition_grid_size` config normalization
- `src/train_configs/local_mac_overfit_video_token_implicit_camera_128_4fps_fast_mac_8192splats_pose_to_plucker.jsonc`
  - matched 128px/4fps/8192-splat config for the new architecture

## Verification

Passed:

- `uv run ruff check src/train/gs_models/implicit_camera.py src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/gs_models/__init__.py src/train/train_video_token_implicit_dynamic.py`
- `uv run python -m py_compile src/train/gs_models/implicit_camera.py src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/gs_models/__init__.py src/train/train_video_token_implicit_dynamic.py`
- direct forward smoke on a tiny `DynamicVideoTokenGSImplicitCameraPoseToPlucker`
  model with shape `[1, 4, 3, 32, 32]`
- config factory smoke for the checked-in pose-to-Plucker JSONC config

No training ablation was run in this chunk, per user instruction not to focus
on the next run yet.
