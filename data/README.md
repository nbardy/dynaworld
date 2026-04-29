# Local Data Roots

Large media artifacts live under `data/` and are intentionally not committed.

See `data/REHYDRATE.md` for the commands and checked-in indexes needed to
re-download or rebuild partial local datasets after deleting raw media.

See `data/CAMERA_CONTRACT.md` for the renderer-facing camera schema and the
current per-dataset camera-adapter status.

## YouTube Motion Ingest

The working root for camera-motion mining is:

```text
data/youtube_motion/
```

The scene-distinct 30-video local pass uses:

```text
data/youtube_scene_distinct/
```

That pass is configured by
`src/dataset_configs/youtube_scene_distinct_30_64_4fps_16f.jsonc` and runs with:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_seed.sh
```

It targets one 64px/4fps/16-frame clip per source video and builds a 20-video
train split plus a 10-video test split. The download stage pulls short source
sections rather than full videos to keep the local disk budget reasonable.

The matched 256px materialization uses the same source videos and split:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_256_seed.sh build-clips
```

It is configured by
`src/dataset_configs/youtube_scene_distinct_30_256_4fps_16f.jsonc` and is the
input for the local 256px V-JEPA baseline configs.

The curated span pass imports older hand-selected YouTube annotations from the
parent repo and appends Dynaworld-local records such as the Matrix bullet-time
test span:

```text
data/youtube_curated_spans/
```

Run it with:

```bash
./src/dataset_scripts/youtube_curated_spans_seed.sh
```

The exact downloaded span MP4s live under `data/youtube_curated_spans/raw/`.
The optional 64px/4fps frame dataset lives under
`data/youtube_curated_spans/clip_sets/`.

Tracked metadata/config:

- `data/youtube_motion/candidates/*.jsonl`
- `data/youtube_multiview_comms/candidates/*.jsonl`
- `data/youtube_multiview_comms/candidates/*.json`

Ignored generated artifacts:

- `data/youtube_motion/raw/`
- `data/youtube_motion/segments/`
- `data/youtube_motion/clip_sets/`
- `data/youtube_motion/logs/`
- `data/youtube_curated_spans/raw/`
- `data/youtube_curated_spans/clip_sets/`
- `data/youtube_curated_spans/logs/`

The final training-compatible output should be a clip set with:

```text
manifest.jsonl
dataset.json
clips/<clip_id>/frames/frame_0000.png
clips/<clip_id>/summary.json
```

## Pseudo-Multicam Practice Sources

Pseudo-multicam practice sources are separate from calibrated multicam
validation. They are useful for cross-view training pressure and qualitative
diagnostics, but not for final held-out-camera metric claims unless calibration
is recovered.

Current routed source:

```text
data/youtube_multiview_comms/
```

That root stores metadata for public production multiview / director-comms
YouTube playlists. Future raw spans, tile crops, and clip sets should follow
the same local-media pattern as the other YouTube roots: keep candidates
tracked, keep downloaded media and generated clips local.

BrettZone/NHRL robot fights belong in the same conceptual category, with a
future root such as:

```text
data/brettzone_nhrl/
```

Those sources provide synchronized fight camera feeds and metadata/API entry
points, but they should be used as practice data unless camera calibration is
added.

TVMCE / Virtual Film Studio is another external pseudo-multicam practice
candidate:

```text
data/external/tvmce/
```

It provides multi-camera TV-show editing data with six synchronized camera
tracks per event and edited-program supervision. Treat it as camera-choice /
cross-view practice data unless a calibration source is found.

## Default Local Mac Clip Set

The default generalization smoke dataset is mined from scene-distinct source
videos:

```text
data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_64_4fps_16f/
data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_256_4fps_16f/
```

It contains:

- 20 train clips from 20 distinct source videos
- 10 test clips from 10 distinct source videos
- 0 source-video overlap between train and test

Build the tiny 64px version with:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_seed.sh
```

Build the matched 256px version with:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_256_seed.sh build-clips
```

The download stage pulls short source sections rather than full videos to keep
the local disk budget reasonable.

Training entrypoints:

- `./src/train_scripts/train_local_mac_30_clip_baseline.sh`
- `./src/train_scripts/train_local_mac_30_clip_vjepa2_256_baseline.sh [local|vjepa|both]`

## Local Multi-Camera Debug Clip Set

The older local multi-camera debug split is:

```text
data/clip_sets/local_mac_30_64_4fps_16f/
```

It contains 30 fixed-length clips sampled at 64px, 4fps, 16 frames:

- 20 train clips from 20 distinct source MP4s
- 10 test clips from 10 distinct source MP4s

Build it from the compact ViVo `athlete_rows` RGB MP4s plus the extracted
Neural 3D Video `coffee_martini` camera MP4s:

```bash
./src/train_scripts/build_local_mac_30_clip_dataset.sh --overwrite
```

The script passes `--max-clips-per-source 1` and `--require-target-count`, so it
fails instead of silently taking multiple windows from the same video. It is
useful for camera-path debugging, but it is not scene-diverse because the source
MP4s are cameras from two local scenes. Generated clip sets stay local and are
ignored by git.

## External Curated Datasets

Curated source datasets that should not be scraped live under:

```text
data/external/
```

The first target is Meta/Facebook Research's Neural 3D Video dataset:

```text
data/external/neural_3d_video/
```

Tracked metadata/config can live beside the dataset root, but raw downloaded
archives and extracted media are ignored:

- `data/external/neural_3d_video/raw/`
- `data/external/neural_3d_video/extracted/`
- `data/external/neural_3d_video/logs/`

ViVo uses the same external-dataset convention:

- `data/external/vivo/raw/`
- `data/external/vivo/extracted/`
- `data/external/vivo/rgb_mp4/`
- `data/external/vivo/logs/`

DeepView Video uses the same convention for raw GT light-field video scenes:

```text
data/external/deepview_video/
```

Run the default seed ingest with:

```bash
./src/dataset_scripts/deepview_video_seed.sh
```

The config registers all 15 official scene archives, about 65.5 GB compressed,
but the default local seed downloads only `03_Dog` and `15_Branches`. `03_Dog`
is currently the usable 4-second validation scene. `15_Branches` is retained as
a tiny structure/calibration smoke scene because it has only 10 frames per
camera. Raw archives, extracted videos, and generated metadata are ignored:

- `data/external/deepview_video/raw/`
- `data/external/deepview_video/extracted/`
- `data/external/deepview_video/logs/`
- `data/external/deepview_video/metadata/`

## Multi-Camera Validation Samples

The first paired source/target validation sample set is:

```text
data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/
```

Build it with:

```bash
./src/dataset_scripts/multicam_val_v1_seed.sh
```

It creates held-out-camera validation pairs rather than independent training
clips:

```text
clips/<sample_id>/summary.json
previews/<sample_id>.mp4
manifest.jsonl
dataset.json
```

Current sources:

- AIST Dance DB refined 10Mbps subset, fixed cameras only.
- Neural 3D Video `coffee_martini`, a non-human dynamic NeRF scene.
- ViVo `athlete_rows`, aligned by capture timestamp overlap.
- DeepView Video `03_Dog`, a 41-camera fisheye light-field scene with
  `models.json` calibration.

## External Pretrained Models

Ex4DGS pretrained release assets are separate from raw validation videos:

```text
data/external/ex4dgs_pretrained/
```

Download and inspect the selected `coffee_martini`, `Birthday`, and `Fabien`
assets with:

```bash
./src/dataset_scripts/ex4dgs_pretrained_val_seed.sh
```

These are checkpoint/model artifacts from the Ex4DGS GitHub release, not
source multi-camera videos. Keep them out of the raw-data validation manifest
until a renderer/evaluator path can consume their checkpoint format directly.

Generated validation frames, previews, logs, AIST CSV metadata, and raw AIST
videos stay local and are ignored by git.

Raw source videos stay at native FPS/resolution. The manifest stores source and
target video paths plus synchronized time windows; metric frames are sampled by
the loader at 128px, 4fps, 16 frames. Side-by-side preview MP4s are rendered
separately at 30fps, so visual review remains smooth while the validation
tensors stay tiny.
