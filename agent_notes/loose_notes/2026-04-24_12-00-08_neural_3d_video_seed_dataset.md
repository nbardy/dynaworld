# Neural 3D Video Seed Dataset

Added a separate curated external-dataset lane for Meta/Facebook Research's
Neural 3D Video dataset instead of mixing it with YouTube scraping.

What changed:

- Added `data/external/README.md`.
- Extended `data/README.md` with `data/external/neural_3d_video/` conventions.
- Added `src/dataset_configs/neural_3d_video_seed.jsonc`.
- Added `src/dataset_pipeline/neural_3d_video.py` with stages:
  - `list-assets`
  - `download`
  - `extract`
  - `inspect`
- Added `src/dataset_scripts/download_neural_3d_video_seed.sh`.
- Ignored large `raw/`, `extracted/`, and `logs/` folders while leaving small metadata visible.

Upstream release metadata:

- Release URL: `https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0`
- Assets:
  - `coffee_martini.zip` ~1.19 GB
  - `cook_spinach.zip` ~1.21 GB
  - `cut_roasted_beef.zip` ~1.14 GB
  - `flame_steak.zip` ~1.20 GB
  - `sear_steak.zip` ~1.19 GB
  - `flame_salmon_1_split.z01` ~1.57 GB
  - `flame_salmon_1_split.z02` ~1.57 GB
  - `flame_salmon_1_split.z03` ~1.57 GB
  - `flame_salmon_1_split.zip` ~0.27 GB

Local data pulled:

- Downloaded `data/external/neural_3d_video/raw/coffee_martini.zip`.
- Extracted to `data/external/neural_3d_video/extracted/coffee_martini/coffee_martini/`.
- Wrote inventory to `data/external/neural_3d_video/metadata/scene_inventory.json`.

Coffee Martini inventory:

- 18 readable cameras.
- `cam00.mp4` is present.
- `poses_bounds.npy` is present.
- Each camera is 300 frames, 30 fps, 10 seconds, 2704x2028.
- Camera IDs have gaps, matching upstream warning that invalid cameras were removed.

Next step:

- Build a converter that can sample this multi-camera scene into a training manifest with camera metadata. This is different from the YouTube single-path clip builder: Neural 3D Video gives synchronized views at each time, so the useful split is context cameras vs held-out `cam00`/other target cameras, not scene-cut mining.
