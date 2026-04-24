# YouTube Motion Dataset Ingest Scaffold

Added a first-pass local dataset ingest scaffold for mining YouTube clips with high camera motion and low scene-cut contamination.

What changed:

- Created `data/youtube_motion/` working subdirectories for candidates, raw downloads, segments, clip sets, and logs.
- Added `data/README.md` to document which pieces are tracked and which media artifacts are ignored.
- Added `src/dataset_configs/youtube_high_camera_motion_seed.jsonc` with initial search tags and thresholds.
- Added `src/dataset_pipeline/youtube_ingest.py` with stages:
  - `search`: calls `yt-dlp` search and writes candidate JSONL.
  - `download`: downloads selected videos into raw media.
  - `segment`: uses OpenCV histogram cut detection and Farneback optical-flow motion scoring, then ffmpeg-extracts high-motion windows.
  - `build-clips`: feeds accepted segment MP4s into existing `src/train/build_clip_dataset.py`.
- Added `src/dataset_scripts/youtube_motion_seed.sh` as the end-to-end seed command.
- Updated `.gitignore` so large downloaded/generated media stays out of git while metadata/config remains trackable.

Validation:

- `.venv/bin/python src/dataset_pipeline/youtube_ingest.py --help` works.
- `.venv/bin/python -m py_compile src/dataset_pipeline/youtube_ingest.py src/train/build_clip_dataset.py` passes.
- Segment stage fails cleanly with `No downloads found. Run the download stage first.`

Current blockers / next work:

- `yt-dlp` is not installed on PATH, so search/download cannot run yet.
- `uv run` is currently blocked by sandbox access to `/Users/nicholasbardy/.cache/uv/sdists-v9/.git`; direct `.venv/bin/python` works.
- Need first real run, threshold calibration, and manual spot-checking before trusting the motion/cut filters.
