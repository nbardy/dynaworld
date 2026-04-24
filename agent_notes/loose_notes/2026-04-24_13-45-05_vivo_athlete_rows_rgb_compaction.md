# ViVo Athlete Rows RGB Compaction

Downloaded one ViVo raw sample scene from the public Google Drive folder:

- Folder path in Drive: `ViVo-Scenes/Data_RawSamples/athlete_rows.zip`
- Local archive path during download: `data/external/vivo/raw/athlete_rows.zip`
- Archive size: about `6.7G`
- Archive contents: `athlete_rows/` with `calibration.json`, `train/`, and `test/`.

Extracted to:

- `data/external/vivo/extracted/athlete_rows/`

Scene inventory after extraction:

- 10 train cameras.
- 4 test cameras.
- 7,010 RGB jpg frames.
- 14,027 depth-related files before cleanup.
- Calibration exists.
- No `rotation_correction.json` in this raw sample.

Compacted RGB to MP4:

- Command: `.venv/bin/python src/dataset_pipeline/vivo.py compact-rgb --config src/dataset_configs/vivo_seed.jsonc --scene athlete_rows`
- Output: `data/external/vivo/rgb_mp4/athlete_rows/`
- 14 MP4 files, one per camera.
- Total MP4 size: about `302M`.
- Sample probe: 2560x1440, 30 fps, 501 frames, 16.7 seconds.
- Manifest: `data/external/vivo/metadata/rgb_mp4_manifest.jsonl`

Deleted depth payload:

- Removed `14,027` `*depth-image*` files from extracted scene.
- Removed the raw `athlete_rows.zip` archive so the depth payload is not kept inside raw storage.
- Remaining extracted scene has `0` depth files, `7,010` RGB jpgs, and `7,012` RGB frame metadata JSON files plus calibration.

Disk after cleanup:

- `data/external/vivo`: `5.8G`
- `data/external/vivo/raw`: `0B`
- `data/external/vivo/extracted`: `5.5G`
- `data/external/vivo/rgb_mp4`: `302M`
- Main filesystem free: about `57Gi`

Important follow-up:

- We did not delete RGB jpgs because the current MP4 manifest does not yet preserve full frame-to-metadata mapping. Before deleting RGB jpgs, create a compact scene manifest that records ordered source RGB filenames, timestamps, camera IDs, and the corresponding MP4 frame indices.
