# Local Data Roots

Large media artifacts live under `data/` and are intentionally not committed.

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

## Default Local Mac Clip Set

The default generalization smoke dataset is mined from scene-distinct source
videos:

```text
data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_64_4fps_16f/
```

It contains:

- 20 train clips from 20 distinct source videos
- 10 test clips from 10 distinct source videos
- 0 source-video overlap between train and test

Build it with:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_seed.sh
```

The download stage pulls short source sections rather than full videos to keep
the local disk budget reasonable.

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
