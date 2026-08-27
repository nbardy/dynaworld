# Deep3D Mask browser jump scene

## Goal

Replace the visually weak tabletop-only browser demo set with one genuinely
dynamic, calibrated multicamera scene while preserving the canonical Python
camera and split contracts. The browser remains a demo adapter rather than a
new trainer or dataset-semantics lane.

## Source and preparation

- Source: official Deep 3D Mask ICCV 2021 evaluation archive
  (`eval_scene.zip`, Google Drive id `1_9KA20cI_0Bs9ERkT65TPtiom3fdQXD0`).
- Official clip: source frames 00670 through 00729, 10 synchronized cameras,
  1920x1080 JPEG, 120 FPS, 60 frames (0.5 seconds), static LLFF poses.
- Scene content: a person jumps beside an outdoor wall and bike racks. This is
  materially more dynamic than the existing cooking/tabletop scenes.
- `src/dataset_pipeline/deep3d_mask.py` validates contiguous frames, cameras
  000-009, and an identical `poses_bounds.npy` in every frame directory. It
  then writes one native-rate MP4 per camera and source metadata.
- The browser-compatible 384x288 stream is intentionally resized to the
  existing 4:3 training contract. The official source is 16:9, so this is a
  compatibility choice rather than native-aspect preservation. A future
  browser-wide 16:9 resolution lane should change the shared resolution
  contract rather than special-case this scene.

## Canonical data contract

- Dataset id: `deep3d_mask`; scene id: `deep3d_mask_eval_jump`.
- Train cameras: cam00-cam04 and cam06-cam09 (9 total).
- Heldout camera: cam05 only; it is excluded from SfM and training.
- Anchor camera: cam04.
- Pose provenance remains distinct as
  `deep3d_mask_llff_opencv_relative_pinhole_v2`, while reusing the same tested
  LLFF-to-anchor-relative OpenCV conversion as Neural 3D Video.
- Browser bundles carry 16 resident atlas frames and page all 60 native frames
  from per-camera MP4 streams at 120 FPS.

## Initialization

Known-pose pycolmap SIFT triangulation ran on source frame 0 at a 1024-pixel
maximum image dimension, using all nine train cameras and no heldout pixels:

- 80,514 keypoints
- 20 requested, matched, and verified camera pairs
- 11,872 raw triangulated points
- 5,517 accepted points
- mean filtered reprojection error: 0.4732 px
- median filtered reprojection error: 0.2853 px
- p90 filtered reprojection error: 1.1385 px

The browser embeds a deterministic farthest-point subset of 4,096 seeds and
marks the initialization `train_only_verified`.

## Browser integration and verification

- Added 96x72 and 384x288 calibrated bundles and a 10-camera temporal stream.
- Added the scene to the top-row selector.
- Fixed a browser validator that incorrectly hard-coded Neural 3D's pose-source
  identity. Validation is now dataset-specific and still fails closed.
- Payload added by the scene is about 25 MiB, dominated by the ten 384 atlases;
  the complete 120 FPS MP4 stream is about 400 KiB.
- Node browser tests: 196/196 pass after the provenance regression test.
- Focused Python data/export tests: 20/20 pass.
- Real Chromium verification: desktop and 390x844 mobile both load the scene,
  show synchronized cam04/cam09/cam05 frames, initialize WebGPU from 4,096
  splats, report the heldout split, and emit zero console errors.

## Remaining scientific caveat

This integration proves data, calibration, initialization, paging, UI, and
runtime compatibility. It does not establish final reconstruction quality on
Deep3D. The 0.5-second clip is excellent for fast dynamic stress testing but is
not a substitute for a longer full-scene Deep3D training run; individual full
compressed scenes are multi-gigabyte downloads and should remain opt-in.
