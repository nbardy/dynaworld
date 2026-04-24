# Local Data Roots

Large media artifacts live under `data/` and are intentionally not committed.

## YouTube Motion Ingest

The working root for camera-motion mining is:

```text
data/youtube_motion/
```

Tracked metadata/config:

- `data/youtube_motion/candidates/*.jsonl`

Ignored generated artifacts:

- `data/youtube_motion/raw/`
- `data/youtube_motion/segments/`
- `data/youtube_motion/clip_sets/`
- `data/youtube_motion/logs/`

The final training-compatible output should be a clip set with:

```text
manifest.jsonl
dataset.json
clips/<clip_id>/frames/frame_0000.png
clips/<clip_id>/summary.json
```

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
