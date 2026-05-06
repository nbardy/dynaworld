# Multicam Val Chunking And Loader Smoke

User asked to split longer multicam clips into chunks when possible, and asked
whether the multicam sources can be used through a unified loader.

Changes made:

- Added config-driven chunk expansion to `src/dataset_pipeline/multicam_val.py`.
  The canonical `multicam_val_v1_128_4fps_16f` config remains single-window by
  default so existing trainer configs keep stable sample IDs and counts.
- Added `src/dataset_configs/multicam_val_v1_chunked_128_4fps_16f.jsonc` and
  `src/dataset_scripts/multicam_val_v1_chunked_seed.sh`.
- Built the generated chunked set at
  `data/multicam_val/clip_sets/multicam_val_v1_chunked_128_4fps_16f/`.
  It currently has 14 samples from 8 base camera pairs:
  AIST 4, Neural 3D Video 4, ViVo 4, DeepView 2.
- Added chunk metadata to manifest records: base starts, per-chunk starts,
  chunk index/count, stride, synchronized available duration, and source/target
  availability.
- Added ViVo calibration/rgb-root metadata to multicam manifest rows so
  `load_multicam_video_bundle(..., rig_init="vivo")` can build cameras.
- Fixed `load_camera_video` to use `target_start_seconds` for pair targets.
  This matters for ViVo because camera MP4s can have different capture-time
  offsets.

Verification:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_multicam_val_chunking.py tests/test_multicam_video_data.py
```

Result: 7 passed.

Runtime loader smokes against the chunked manifest:

- AIST chunk loaded with `pose_source='aist_plusplus_relative_pinhole'`.
- ViVo chunk loaded with `pose_source='vivo_calibration_relative_pinhole'` and
  asymmetric starts `source_start_seconds=5.0`,
  `target_start_seconds=11.917075`.
- Neural 3D Video chunk loaded with
  `pose_source='neural_3d_llff_relative_pinhole'`.
- DeepView chunk loaded with `pose_source='deepview_models_relative_pinhole'`.

Status command:

```bash
./src/dataset_scripts/local_data_status.sh
```

Reported the canonical set at 8 clips and the chunked set at 14 clips.

Remaining caveat:

- The unified loader is real for curated calibrated/semicalibrated multicam
  validation manifests: AIST, DeepView, Neural 3D Video, and ViVo. YouTube
  playlist/comms and BrettZone are still a separate scalable pseudo-multicam
  practice lane; they need their own manifest schema/adapter before they should
  be treated as the same calibrated loader surface.
