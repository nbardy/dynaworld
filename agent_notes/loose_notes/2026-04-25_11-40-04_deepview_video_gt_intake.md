# DeepView Video GT Intake

User pointed to `https://github.com/augmentedperception/deepview_video_dataset`
as the real GT validation dataset with 15 scenes.

README facts:

- It contains raw multi-camera light-field video recordings from the DeepView
  Video paper.
- There are 15 scene zip archives.
- Each scene contains up to 46 synchronized camera videos, with a few missing
  cameras in some scenes.
- Each archive has `models.json` calibration from the authors' structure-from-
  motion solver.
- Cameras are Yi4k action-camera fisheye videos, so fisheye projection and
  radial distortion need to stay in the manifest.

Measured archive sizes via HTTP HEAD:

- Full configured set is about 65.46 GB compressed.
- `15_Branches` is tiny at about 0.18 GB, but after extraction it has only 10
  frames per camera, about 0.33s at 29.97fps. It is useful for structure smoke
  only, not for the current 16-frame/4fps validation window.
- `03_Dog` is about 1.40 GB compressed and is usable locally: 41 calibrated
  fisheye videos, 2560x1920, 150 frames, about 5.0s at 29.97fps.

Added DeepView intake:

- `src/dataset_configs/deepview_video_seed.jsonc`
- `src/dataset_scripts/deepview_video_seed.sh`
- `src/dataset_pipeline/deepview_video.py`

Downloaded/extracted local seed scenes:

- `03_Dog`
- `15_Branches`

Integrated DeepView into the existing paired validation manifest:

- `deepview_03_Dog_camera_0001_to_camera_0015`
- `deepview_03_Dog_camera_0001_to_camera_0040`

Current `multicam_val_v1_128_4fps_16f` now has 8 paired GT validation samples:
2 AIST, 2 Neural 3D Video, 2 ViVo, and 2 DeepView. Loader smoke passed for all
8 records with source/target tensors shaped `(16, 3, 128, 128)`.
