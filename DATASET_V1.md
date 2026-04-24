# DATASET_V1

This is the first dataset contract for Dynaworld baselines. It separates cheap
local smoke data from real novel-view validation data.

The core rule: source videos are useful for fast training/debugging, but
synchronized multi-camera sequences are required for validation accuracy.

## Current Local Smoke Data

The current local Mac default should stay small and fast:

- 30 scene-distinct source videos.
- 20 train clips and 10 test clips.
- One clip per source video.
- No train/test source-video overlap.
- 64px, 4fps, 16 frames for tiny local runs.

This is for fast baseline iteration on a Mac, not for final validation. It
checks that the trainer, dataloader, renderer path, and loss plumbing work on
different scenes without overfitting one camera rig or one semantic setting.

The older local multi-camera debug split remains useful for camera-path
debugging, but it is not the default generalization dataset because its MP4s
come from cameras around only a small number of scenes.

## Hand-Labeled YouTube Spans

The hand-labeled YouTube annotations found so far live in the parent repo:

```text
/Users/nicholasbardy/git/gsplats_browser/corrective_splat_trainer/manual_list.jsonl
```

That discovered file was imported into the Dynaworld curated-span pipeline and
expanded into 27 spans:

- 26 train spans from the old hand-labeled file.
- 1 test span for the Matrix bullet-time clip:

```text
https://www.youtube.com/watch?v=8DajVKAkL50
0:37 => 0:46
title: matrix bullettime
```

Download status:

- 26 exact-span MP4s downloaded.
- 1 old train record failed because the annotated timestamp is now outside the
  currently resolved YouTube duration.
- The optional 64px/4fps/16-frame build produced 19 usable clips; the very short
  2-3 second spans downloaded as MP4s but are too short for 16 frames at 4fps.

This covers the hand-labeled file that was discovered. If there are other
hand-labeled lists elsewhere, they still need to be found and imported.

These spans are valuable for local qualitative checks and special-effect
coverage, but they do not provide held-out novel-camera ground truth.

## Why Multi-Camera Validation Is Required

For validation we need a task with a real target image:

```text
original timeframe + source camera(s) -> held-out novel camera, same timeframe
```

A monocular YouTube clip cannot answer whether a generated novel angle is
correct. It can only tell us whether the output looks plausible. That is not
enough for model selection.

A synchronized multi-camera dataset lets us:

- Feed frames from one camera or a small source-camera set.
- Query a different camera angle at the same frame indices.
- Compare the model output against the actual held-out camera frames.
- Report SSIM, PSNR, LPIPS, and frame-distribution FID.
- Later add FVD or temporal consistency metrics once the frame metrics are
  stable.

## AIST Dance DB Validation Track

AIST Dance DB is a strong first validation target:

- Official site: <https://aistdancedb.ongaaccel.jp/>
- Database structure: <https://aistdancedb.ongaaccel.jp/database_structure/>
- Data formats and naming rules: <https://aistdancedb.ongaaccel.jp/data_formats/>

Useful properties from the official docs:

- It is a street-dance video database with many genres, dancers, and cameras.
- The database structure page reports 1,618 dances and 13,940 videos.
- The data format page defines fixed cameras `c01-c09` and moving camera `c10`.
- Filenames encode genre, situation, camera, dancer, music, and choreography
  identifiers, for example `gBR_sBA_c01_d03_mBR3_ch04.mp4`.

Initial use:

- Start with fixed cameras `c01-c09`.
- Exclude moving camera `c10` from the first validation benchmark.
- Treat `c10` as a later stress split once fixed-camera validation is stable.
- Use the official access path and terms of use; do not scrape around the
  database.

## Validation Contract

The unit of validation is a synchronized multi-view window:

```json
{
  "dataset": "aist_dance_db",
  "sequence_id": "gBR_sBA_d03_mBR3_ch04",
  "frame_start": 0,
  "frame_count": 16,
  "fps": 4,
  "input_cameras": ["c01"],
  "target_cameras": ["c05", "c09"],
  "split": "val"
}
```

The exact `sequence_id` format can change after intake, but it must identify the
same dance/time/motion across all cameras. Camera ID must not be the sequence
identity.

For each validation item:

1. Load the source-camera frames for the requested timeframe.
2. Build the model world state from those source frames.
3. Query each held-out target camera at the same frame indices.
4. Compare predicted frames against the real target-camera frames.
5. Aggregate metrics by sequence, target camera, camera gap, genre, and motion
   type.

Strict rule: target frames must be the same timestamps as the input window, not
nearby frames and not a separate performance.

## Metrics

Minimum metrics for DATASET_V1:

- SSIM per frame, averaged per sequence and per target camera.
- PSNR per frame, averaged the same way.
- LPIPS per frame when the dependency is available locally.
- FID over sampled predicted/target frames for each validation split.

Report these separately:

- Same-sequence held-out camera accuracy.
- Cross-sequence generalization.
- Camera-gap buckets, such as near, side, rear, and low camera.

Do not collapse all validation into one scalar until the per-camera report is
available. A model can look good from adjacent cameras while failing on rear or
low-angle views.

## Split Policy

There are two separate splits:

1. Training/generalization split: separate sequences, dancers, music/choreo
   IDs, and scenes across train/test when possible.
2. Novel-camera validation split: hold out cameras within validation sequences
   so there is a same-time ground-truth target.

For AIST, split by sequence identity first, then choose camera roles inside each
sequence. Do not put `c01` of a sequence in train and `c05` of the same sequence
in test if the goal is cross-sequence generalization. That would leak motion,
dancer, lighting, clothing, and timing.

A first small split can be:

- 20 train sequences from fixed cameras.
- 10 validation sequences from different fixed-camera sequences.
- Input camera: `c01`.
- Held-out target cameras: `c03`, `c05`, `c07`, `c09`.
- Tiny local resolution/fps/window: 64px, 4fps, 16 frames.

After the loader is correct, add:

- Multi-source input cameras, such as `c01+c05`.
- Larger windows, such as 32 or 64 frames.
- Higher resolution, such as 128px or 256px.
- Moving-camera `c10` as a separate stress benchmark.

## Intake Milestone

The next concrete milestone is an AIST manifest builder that writes a normalized
multi-camera manifest without committing raw media:

```text
data/external/aist_dance_db/
  manifests/
    sequences.jsonl
    windows_64_4fps_16f.jsonl
    dataset.json
  raw/        # ignored
  extracted/  # ignored
  clip_sets/  # ignored
  logs/       # ignored
```

Each manifest row should preserve:

- Source URL or official download identifier.
- Original filename.
- Parsed genre, situation, camera, dancer, music, and choreography IDs.
- Sequence ID with camera removed.
- Camera ID.
- Duration, fps, width, height, frame count.
- Split assignment.

The first benchmark should be:

```text
input:  AIST c01, same sequence/time window
target: AIST c05, same sequence/time window
metric: SSIM + PSNR + LPIPS if available + frame FID
```

That gives us a real validation target for "original timeframe in, novel angle
same timeframe out" instead of judging novel views only by visual plausibility.
