# CamXTime OpenGL Pose Fix

## Context

A deeper rig diagnostic pass on the CamXTime row-one smoke exposed a real pose
convention bug. The visualizer initially drew the 120-camera full-grid rig in
the trainer frame, but the numeric optical-axis fit placed the common focus
behind every camera's `c2w[:, 2]` axis.

That is incompatible with DynaWorld's `CameraSpec` contract: camera-frame `+Z`
is the optical axis, and the renderer keeps points with positive camera-space
depth.

## Finding

CamXTime full-grid `camera_data.json` stores Blender/OpenGL-style camera poses:

- `+X` is camera right
- `+Y` is image up
- `-Z` is the view direction

DynaWorld uses OpenCV-style camera axes:

- `+X` is camera right
- `+Y` is image down
- `+Z` is the view direction

So the adapter must flip the camera-frame Y and Z columns after reading or
inverting the CamXTime matrix:

```text
c2w_opencv = c2w_opengl @ diag([1, -1, -1, 1])
```

## Changes

- `src/train/multicam_video_data.py` now treats CamXTime poses as OpenGL by
  default and converts them into the renderer's OpenCV/+Z convention.
- CamXTime records can override this with `camxtime_camera_convention:
  "opencv"` for already-converted fixtures.
- The row-one smoke manifest now explicitly records
  `camxtime_camera_convention: "opengl"` and reports
  `camera_model: "camxtime_pinhole_opengl_c2w"`.
- `src/dataset_scripts/visualize_multicam_rig.py` now supports
  `--all-camxtime-role train_except_heldout`, which models the intended
  "one input, many supervised target cameras, keep a heldout subset" split.
- The visualizer now emits numeric pose diagnostics: rotation determinant,
  orthogonality, camera radius stats, train/heldout counts, condition identity
  error, and a least-squares optical-axis focus point with per-camera miss and
  behind-focus checks.

## Verification

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_multicam_video_data.py -q
```

Result: `10 passed`.

```bash
rtk env PYTHONPATH=src/train .venv/bin/python src/dataset_scripts/visualize_multicam_rig.py \
  --config src/train_configs/local_mac_camxtime_row1_full_grid_multicam_smoke.jsonc \
  --output outputs/camera_rigs/camxtime_scene1_all_cameras_train_except_heldout.html \
  --json-output outputs/camera_rigs/camxtime_scene1_all_cameras_train_except_heldout.json \
  --input-camera camera_000 \
  --all-camxtime-cameras \
  --all-camxtime-role train_except_heldout \
  --frame-count 1
```

Result: 120 cameras, 119 train / 1 heldout, pose source
`camxtime_full_grid_opengl_to_opencv_relative_pinhole`, focus point at roughly
`[0, 0, +4.03]` in the input-camera frame, and no cameras with focus behind.

```bash
rtk env PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_camxtime_row1_full_grid_multicam_smoke.jsonc
```

Result: the 1-step smoke completed using the corrected pose source and rendered
train plus heldout validation. Offline W&B run:
`wandb/offline-run-20260519_001344-hdwqoj6a`.

## Remaining Work

- The checked-in CamXTime fixture still only has three MP4s. Full
  one-input/many-loss training needs the rest of the `camera_*.mp4` files or a
  packed-archive extractor/downloader.
- The current external-rig trainer eagerly loads every configured train camera
  video. A real 120-camera run should add a camera-target sampler or streaming
  bundle path instead of loading all target videos into memory at once.
- The current trainer requires a heldout split. For "train on literally all 120
  cameras" with no heldout, we would need to relax the split validation and
  handle `heldout_frames=None` through the bundle/adapters. For research, a
  cleaner default is train on 119 and keep one or more cameras held out.
