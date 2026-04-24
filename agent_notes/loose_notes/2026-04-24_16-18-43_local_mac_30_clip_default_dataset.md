# Local Mac 30 Clip Default Dataset

Goal:

- Make the small local baseline dataset concrete: 30 short clips, with 20 train and 10 test.
- Keep it fast enough for Mac local smoke runs.

Data source:

- Used the compact ViVo `athlete_rows` RGB MP4s already present under:
  - `data/external/vivo/rgb_mp4/athlete_rows/train`
  - `data/external/vivo/rgb_mp4/athlete_rows/test`
- Preserved ViVo's source split when generating the derived clip set.

Code changes:

- Extended `src/train/build_clip_dataset.py` with split-aware inputs:
  - `--train-input`
  - `--test-input`
  - `--train-count`
  - `--test-count`
  - `--source-schedule round_robin`
- Kept the old `--input` mode working as a train-only build.
- Added split manifests:
  - `manifest.jsonl`
  - `train_manifest.jsonl`
  - `test_manifest.jsonl`
- Added manifest-backed loading to `src/train/train_video_token_implicit_dynamic.py` so the video-token implicit-camera trainer can sample across train clips and optionally load test clips for eval.
- Added default scripts:
  - `src/train_scripts/build_local_mac_30_clip_dataset.sh`
  - `src/train_scripts/train_local_mac_30_clip_baseline.sh`
- Added default config:
  - `src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc`

Generated local artifact:

- `data/clip_sets/local_mac_30_64_4fps_16f/`
- 30 clips total.
- 20 train clips from 20 distinct source MP4s.
- 10 test clips from 10 distinct source MP4s.
- 0 source-path overlap between train and test.
- 16 frames per clip.
- 4 fps.
- 64px extracted frames.
- 480 PNG frames total.
- `data/clip_sets/` is ignored because these are generated local artifacts.

Follow-up change:

- The first pass accidentally allowed 20 train clips from only 10 ViVo train-camera MP4s. The default script now mixes local sources to enforce the user's intended source count:
  - train: 10 ViVo `athlete_rows/train` MP4s + 10 Neural 3D Video `coffee_martini` camera MP4s
  - test: 4 ViVo `athlete_rows/test` MP4s + 6 remaining Neural 3D Video `coffee_martini` camera MP4s
- `src/train/build_clip_dataset.py` now supports `--max-clips-per-source` and `--require-target-count`; the default local script uses `--max-clips-per-source 1`, so the 20/10 counts are source counts too.

Validation:

- `uv run python -m py_compile src/train/build_clip_dataset.py src/train/train_video_token_implicit_dynamic.py`
- `./src/train_scripts/build_local_mac_30_clip_dataset.sh --dry-run`
- `./src/train_scripts/build_local_mac_30_clip_dataset.sh --overwrite`
- `wc -l` confirmed `30/20/10` lines for all/train/test manifests.
- `find ... -name 'frame_*.png' | wc -l` confirmed `480` extracted frames.
- Ran one offline MPS training step against `src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc` after setting `train.steps = 1`; it loaded 20 train sequences and 2 eval sequences and completed the step.
- Ran a second one-step offline MPS smoke with `video_log_every = 1` and `eval_max_sequences = 1`; it rendered/logged validation video payloads from a held-out test clip successfully.
- After the distinct-source patch, dry-run printed `sources={'train': 20, 'test': 10}`, `dataset.json` recorded `train_source_count=20` and `test_source_count=10`, a manifest check found `overlap 0`, and another one-step offline MPS smoke passed.

Notes:

- The repo already had unrelated dirty work in architecture docs, V-JEPA config/model files, key learnings, and the fast-mac submodule. Those were not part of this dataset pass.
- `uv` printed warnings about the parent `gsplats_browser/pyproject.toml` not containing a `project` table. The commands still exited successfully.

Second follow-up: scene-diverse source videos

- The user clarified that different camera MP4s from the same scene are not enough; train/test needs different videos from different scenes for generalization.
- Added `src/dataset_configs/youtube_scene_distinct_30_64_4fps_16f.jsonc` and `src/dataset_scripts/youtube_scene_distinct_30_seed.sh`.
- Updated `src/dataset_pipeline/youtube_ingest.py` to:
  - run `yt-dlp` through `uv run --with yt-dlp` / `python -m yt_dlp` when the binary is not globally installed
  - continue past individual download/segment failures when configured
  - download short source sections instead of whole videos
  - build train/test manifests directly with one clip per source
- The first full-video download attempt was stopped after a few candidates because it consumed multiple GB. The raw folder from that attempt was deleted and regenerated using 8-second source sections.
- Built `data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_64_4fps_16f/`:
  - 30 clips total
  - 20 train clips from 20 source videos
  - 10 test clips from 10 source videos
  - 0 train/test source overlap
  - 480 extracted PNG frames
  - total `data/youtube_scene_distinct` size about `87M`
- Updated `src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc` and `src/train_scripts/train_local_mac_30_clip_baseline.sh` so the default tiny baseline now uses the scene-distinct mined dataset instead of the local two-scene camera debug set.
- One-step offline MPS smoke with eval-video logging passed against the new default scene-distinct dataset.
