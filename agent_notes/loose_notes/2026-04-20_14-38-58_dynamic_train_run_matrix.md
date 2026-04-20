# Dynamic Train Run Matrix

User asked which three Dynaworld dynamic train scripts are current and which
configs to run for stability checks.

Confirmed current dynamic trainer entrypoints:

- Known/prebaked camera: `src/train/dynamicTokenGS.py`
- Image-encoder implicit camera: `src/train/train_camera_implicit_dynamic.py`
- Video-token implicit camera: `src/train/train_video_token_implicit_dynamic.py`

Confirmed current shell wrappers:

- `src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh`
- `src/train_scripts/train_full_dynamic_with_implicit_camera_all_frames.sh`
- `src/train_scripts/train_full_dynamic_with_video_token_implicit_camera_all_frames.sh`

Compatibility/extra wrappers:

- `src/train_scripts/train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh`
  is an alias to the implicit-camera wrapper.
- `src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh`
  runs the video-token trainer with the short smoke config.
- `src/train/train_camera_implict_dynamic.py` is a typo shim; prefer the correctly
  spelled `train_camera_implicit_dynamic.py`.

Default config mapping:

- Prebaked: `src/train_configs/local_mac_overfit_prebaked_camera.jsonc`
- Prebaked tiled renderer variant: `src/train_configs/local_mac_overfit_prebaked_camera_tiled.jsonc`
- Image implicit: `src/train_configs/local_mac_overfit_image_implicit_camera.jsonc`
- Video-token smoke: `src/train_configs/local_mac_overfit_video_token_smoke.jsonc`
- Video-token full: `src/train_configs/local_mac_overfit_video_token_full.jsonc`

Checks run in this session:

- `uv run python -m py_compile ...` across current train modules: passed.
- `bash -n` across current train shell wrappers: passed.
- Config load/resolve for prebaked, prebaked tiled, image implicit, video-token
  smoke, and video-token full: passed.
- Required local assets were present:
  `test_data/test_video_384_3fps.mp4`,
  `test_data/test_video_small.mp4`,
  `test_data/dust3r_outputs/test_video_small_all_frames/summary.json`, and
  `test_data/dust3r_outputs/test_video_small_all_frames/per_frame_cameras.json`.
