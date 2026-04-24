# External Datasets

This folder is for curated third-party datasets with stable source releases.

It is separate from `data/youtube_motion/` because these datasets already have
known capture structure, licenses, and metadata. Do not run YouTube search or
scene mining against this folder.

## Neural 3D Video

Source: <https://github.com/facebookresearch/Neural_3D_Video>

The upstream repo is archived and hosts release assets for the CVPR 2022
dataset "Neural 3D Video Synthesis from Multi-View Video". Each extracted
sequence contains synchronized multi-camera videos named like `cam00.mp4`,
`cam01.mp4`, etc., plus `poses_bounds.npy`.

`cam00.mp4` is the upstream held-out center reference camera. The valid camera
IDs can have gaps because upstream filtered invalid or unsynchronized streams.

Current seed config:

```bash
.venv/bin/python src/dataset_pipeline/neural_3d_video.py list-assets --config src/dataset_configs/neural_3d_video_seed.jsonc
.venv/bin/python src/dataset_pipeline/neural_3d_video.py download --config src/dataset_configs/neural_3d_video_seed.jsonc
.venv/bin/python src/dataset_pipeline/neural_3d_video.py extract --config src/dataset_configs/neural_3d_video_seed.jsonc
.venv/bin/python src/dataset_pipeline/neural_3d_video.py inspect --config src/dataset_configs/neural_3d_video_seed.jsonc
```

The first local seed is `coffee_martini.zip`, which is about `1.19 GB`
compressed and about `1.1 GB` extracted on disk.

Release assets seen on `v1.0`:

- `coffee_martini.zip`
- `cook_spinach.zip`
- `cut_roasted_beef.zip`
- `flame_steak.zip`
- `sear_steak.zip`
- `flame_salmon_1_split.z01`
- `flame_salmon_1_split.z02`
- `flame_salmon_1_split.z03`
- `flame_salmon_1_split.zip`

## ViVo

Source: <https://vivo-bvicr.github.io/>

ViVo is a newer RGB-D multi-view volumetric-video dataset. It is not a simple
GitHub-release dataset:

- The website says downloads require an MS Form submission and email link.
- A temporary Google Drive folder is linked from the project page.
- The catalogue warns that the full zipped dataset is `> 1 TB`.
- Each scene has `14x` RGB and depth video-camera pairs.
- The documented paper split uses `10x` train cameras and `4x` test cameras.
- Raw data includes per-frame RGB, depth, intrinsics/extrinsics metadata.

Use this dataset selectively. Start with one manually downloaded scene, then
inspect it locally:

```bash
.venv/bin/python src/dataset_pipeline/vivo.py inspect --config src/dataset_configs/vivo_seed.jsonc
```

The first converter should target a tiny RGB-only subset before expanding to
depth, masks, or generated point clouds.

If a manually downloaded scene is too large, compact it to RGB MP4s first:

```bash
.venv/bin/python src/dataset_pipeline/vivo.py compact-rgb --config src/dataset_configs/vivo_seed.jsonc --scene <scene_name>
```

Only after verifying the MP4 outputs, the heavy depth/pointcloud/mask folders
can be removed with the explicit destructive flag:

```bash
.venv/bin/python src/dataset_pipeline/vivo.py compact-rgb --config src/dataset_configs/vivo_seed.jsonc --scene <scene_name> --delete-heavy
```

For `athlete_rows`, RGB images and MP4s are `2560x1440` at metadata-declared
`30 fps`. The compactor now writes:

- `data/external/vivo/metadata/rgb_mp4_manifest.jsonl` with per-camera
  resolution, encoded FPS, frame counts, metadata counts, and timestamp-derived
  FPS.
- `data/external/vivo/metadata/rgb_mp4_frames/<scene>/<split>/<camera>.json`
  with ordered RGB filename to MP4-frame-index records.

The first `athlete_rows` pull has two imperfect cameras: `train/000409113112`
has `498` RGB frames vs `499` RGB metadata entries and two RGB frames without
matching metadata; `train/000951614712` has `500` RGB frames vs `501` metadata
entries. The other 12 cameras have `501` RGB frames and matching metadata.
