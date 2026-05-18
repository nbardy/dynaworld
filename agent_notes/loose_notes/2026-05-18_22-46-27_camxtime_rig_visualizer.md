# CamXTime Rig Visualizer

## Context

The CamXTime row-one smoke path can load and train on a small three-camera slice, but we need to reason about the larger camera grid before treating held-out camera supervision as real novel-view training. The important contract is: encode one input camera video, then render/query from other calibrated camera poses and put reconstruction losses on those target views.

## What Changed

- Added `src/dataset_scripts/visualize_multicam_rig.py`, a raw dataset-camera diagnostic that reuses the existing multicam camera adapters instead of inventing separate pose math.
- The script accepts a multicam train config and writes:
  - a self-contained HTML canvas view with camera frustums drawn as pyramids
  - a JSON pose dump containing centers, axes, intrinsics, `camera_to_world`, and `world_to_camera`
- It supports the current `rig_init` adapters: `camxtime`, `deepview`, `aist`, `neural_3d_video`, `vivo`, and `orthogonal_origin`.
- For CamXTime, `--all-camxtime-cameras` draws every pose from `camera_data.json` while marking the input camera as the single train/condition view and all other cameras as heldout for geometry inspection.

## Verification

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile src/dataset_scripts/visualize_multicam_rig.py
rtk env PYTHONPATH=src/train .venv/bin/python src/dataset_scripts/visualize_multicam_rig.py \
  --config src/train_configs/local_mac_camxtime_row1_full_grid_multicam_smoke.jsonc \
  --output outputs/camera_rigs/camxtime_scene1_all_cameras.html \
  --json-output outputs/camera_rigs/camxtime_scene1_all_cameras.json \
  --input-camera camera_000 \
  --all-camxtime-cameras \
  --frame-count 1
```

Result: the diagnostic wrote a 120-camera CamXTime rig in `anchor_relative_opencv_plus_z_forward`, with `camera_000` as the input/condition camera and 119 heldout/query cameras. The generated pose files are intentionally under `outputs/` and ignored by git.

## Notes

- The existing checked-in CamXTime smoke manifest still names only `camera_000`, `camera_020`, and `camera_040` because only the row-one video subset was extracted for training smoke.
- Full one-input/many-loss training needs the rest of the CamXTime MP4s present or a downloader/extractor that materializes the requested camera videos from the packed archive.
- The pose math is already anchor-relative: for each target view, `rel_w2c = inv(c2w_target) @ c2w_anchor`. If the anchor equals the condition camera, the input video is the origin frame and each camera query is relative to that input.
