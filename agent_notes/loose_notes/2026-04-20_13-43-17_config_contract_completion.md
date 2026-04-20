# Config Contract Completion

This note closes the cleanup items that were still open after the JSONC and
viewport-render pass.

## User Request

The user pointed at the remaining warning rows:

- make every trainer JSONC/config-based
- fully remove the dict-style `SequenceData[...]` bridge
- replace tuple model outputs with named payloads

## Implementation

Added `src/train/config_utils.py` as the shared JSONC utility:

- `strip_jsonc_comments(...)`
- `load_config_file(...)`
- `resolved_config(...)`
- `serialize_config_value(...)`

Removed `SequenceData.__getitem__` from `src/train/runtime_types.py`. Trainer
code now reads attributes such as `sequence_data.frames`,
`sequence_data.frame_times`, and `sequence_data.cameras` directly.

Converted trainer-facing model forwards to return `GaussianSequence`:

- `DynamicTokenGS`
- `DynamicTokenGSImplicitCamera`
- `DynamicVideoTokenGSImplicitCamera`
- `TokenGS`

Implicit-camera outputs now carry `CameraState` on `GaussianSequence.camera_state`
instead of leaking dict/tuple payloads into trainers.

Moved trainer entrypoints to JSONC configs:

- `src/train_configs/local_mac_overfit_single_image.jsonc`
- `src/train_configs/local_mac_overfit_single_image_tiled.jsonc`
- `src/train_configs/local_mac_overfit_prebaked_camera.jsonc`
- `src/train_configs/local_mac_overfit_prebaked_camera_tiled.jsonc`
- `src/train_configs/local_mac_overfit_image_implicit_camera.jsonc`
- `src/train_configs/local_mac_overfit_video_token_smoke.jsonc`
- `src/train_configs/local_mac_overfit_video_token_full.jsonc`

The shell wrappers now choose config files and call the Python trainer. The
remaining `argparse` hit under `src/train/` is `run_dust3r_video.py`, which is a
camera-prebake utility rather than a trainer.

## Verification

Static checks:

```bash
uv run python -m py_compile src/train/config_utils.py src/train/runtime_types.py src/train/rendering.py src/train/train_logging.py src/train/gs_models/blocks.py src/train/gs_models/token_gs.py src/train/gs_models/dynamic_token_gs.py src/train/gs_models/dynamic_token_gs_implicit_camera.py src/train/gs_models/dynamic_video_token_gs_implicit_camera.py src/train/tokenGS.py src/train/tokenGS_tiled.py src/train/dynamicTokenGS.py src/train/dynamicTokenGS_tiled.py src/train/train_camera_implicit_dynamic.py src/train/train_camera_implict_dynamic.py src/train/train_image_encoder_implicit_camera_baseline.py src/train/train_video_token_implicit_dynamic.py
for f in src/train_scripts/train_full_dynamic_with_camera_prebake_all_frames.sh src/train_scripts/train_full_dynamic_with_image_encoder_implicit_camera_baseline.sh src/train_scripts/train_full_dynamic_with_implicit_camera_all_frames.sh src/train_scripts/train_full_dynamic_with_video_token_implicit_camera_all_frames.sh src/train_scripts/train_smoke_dynamic_with_video_token_implicit_camera.sh; do bash -n "$f"; done
```

Config load check covered all checked-in trainer JSONC files.

One-step W&B smokes from this pass:

- `local-mac-overfit-single-image-1step-contracts`: W&B run `dr4oq2ch`, loss about `0.2036`
- `local-mac-overfit-single-image-tiled-1step-contracts`: W&B run `iunbfd37`, loss about `0.2010`

One-step W&B smokes from the immediately preceding contract migration, before
this note was written:

- `local-mac-overfit-prebaked-camera-1step-contracts`: W&B run `gotkdizp`, loss about `0.2002`
- `local-mac-overfit-image-implicit-camera-1step-contracts`: W&B run `cse3cnmr`, loss about `0.4554`
- `local-mac-overfit-video-token-smoke-1step-contracts`: W&B run `kfa41vye`, loss about `0.4616`
- `local-mac-overfit-video-token-full-1step-contracts`: W&B run `6w00e77z`, loss about `0.5025`

## Remaining Follow-Up

The main remaining cleanup is structural, not contract-level: the known-camera
and image-implicit trainers are still procedural while the video-token trainer
has a `Trainer` class. Keep them separate, but consider moving each procedural
loop into its own small trainer class if the files keep growing.
