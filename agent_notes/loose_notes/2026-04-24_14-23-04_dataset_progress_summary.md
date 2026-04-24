# Dataset Progress Summary

Current local dataset state after setting up YouTube, Neural 3D Video, and ViVo ingest lanes.

## Disk Budget

- Main filesystem after cleanup: about `57Gi` free.
- `data/external/neural_3d_video`: about `2.2G`.
- `data/external/vivo`: about `5.8G`.
- `data/youtube_motion`: still empty.

Important operational conclusion:

- Do not bulk-download multi-view datasets locally.
- Use one scene at a time, compact to MP4/manifests, and delete raw archives/depth payloads after verifying outputs.
- ViVo full dataset is not locally viable (`>1TB` zipped according to project docs).

## Neural 3D Video

Source:

- `https://github.com/facebookresearch/Neural_3D_Video`
- Release tag queried: `v1.0`.

Local data:

- Downloaded `coffee_martini.zip`.
- Extracted scene path: `data/external/neural_3d_video/extracted/coffee_martini/coffee_martini/`.
- Raw archive remains in `data/external/neural_3d_video/raw/coffee_martini.zip`.
- Scene inventory: `data/external/neural_3d_video/metadata/scene_inventory.json`.

Inventory:

- 18 readable camera MP4s.
- `cam00.mp4` present.
- `poses_bounds.npy` present.
- Each camera is 300 frames, 30 fps, 10 seconds, 2704x2028.

Next step:

- Build a small multi-camera training/eval manifest from this scene before downloading more Neural 3D Video scenes.

## ViVo

Source:

- Project page: `https://vivo-bvicr.github.io/`
- Public Drive folder used: `ViVo-Scenes/Data_RawSamples/athlete_rows.zip`.

Downloaded/extracted:

- Downloaded `athlete_rows.zip` via browser/Drive confirmation flow.
- Archive was about `6.7G`.
- Extracted to `data/external/vivo/extracted/athlete_rows/`.
- Deleted the raw archive after compacting/depth cleanup.

Raw extracted scene before cleanup:

- 10 train cameras.
- 4 test cameras.
- 7,010 RGB jpgs.
- 14,027 depth-related files.
- `calibration.json` present.
- No `rotation_correction.json` in this raw sample.

Compacted RGB:

- Command used: `.venv/bin/python src/dataset_pipeline/vivo.py compact-rgb --config src/dataset_configs/vivo_seed.jsonc --scene athlete_rows`.
- Output MP4 root: `data/external/vivo/rgb_mp4/athlete_rows/`.
- Produced 14 MP4s, one per camera.
- MP4 total size: about `302M`.
- Source/MP4 resolution: `2560x1440`.
- Encoded FPS: `30`.
- Typical duration: about `16.7s`.

Metadata/manifests:

- Summary manifest: `data/external/vivo/metadata/rgb_mp4_manifest.jsonl`.
- Per-camera ordered frame manifests: `data/external/vivo/metadata/rgb_mp4_frames/athlete_rows/<split>/<camera>.json`.
- These now record per-camera resolution, encoded FPS, metadata-declared FPS, timestamp-derived FPS, RGB frame counts, metadata counts, and RGB filename to MP4 frame index mappings.

Cleanup:

- Deleted all `*depth-image*` files from extracted `athlete_rows`.
- Deleted the raw `athlete_rows.zip`.
- Kept RGB jpgs and RGB per-frame metadata for now.

Alignment caveats:

- All 14 cameras are `2560x1440` and encoded at `30 fps`.
- Frame counts are not perfectly uniform:
  - 12 cameras have `501` RGB frames with matching RGB metadata.
  - `train/000409113112` has `498` RGB frames, `499` RGB metadata files, and 2 RGB frames without matching metadata.
  - `train/000951614712` has `500` RGB frames and `501` RGB metadata files.
- Do not assume all ViVo cameras are exactly frame-aligned by index. Use the per-camera manifests and timestamps.

Next steps:

- Create a compact training/eval manifest format for ViVo that chooses aligned frame subsets across cameras using timestamps.
- Once the compact manifest is proven sufficient, optionally delete RGB jpgs and keep only MP4s + metadata/manifests to reduce ViVo from about `5.8G` toward `302M` plus metadata.

## YouTube Motion Miner

Scaffold exists:

- Config: `src/dataset_configs/youtube_high_camera_motion_seed.jsonc`.
- Pipeline: `src/dataset_pipeline/youtube_ingest.py`.
- Script: `src/dataset_scripts/youtube_motion_seed.sh`.
- Data root: `data/youtube_motion/`.

Status:

- No YouTube data downloaded yet.
- `yt-dlp` was not found on PATH during setup.
- `ffmpeg` is available.

Next step:

- Install or expose `yt-dlp`, then run search/download/segment/build-clips in small batches.
