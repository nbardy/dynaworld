# DATASET_V1

This is the first dataset contract for Dynaworld baselines. It separates cheap
local smoke data from real novel-view validation data.

The core rule: source videos are useful for fast training/debugging, but
synchronized multi-camera sequences are required for validation accuracy.

## Current Dataset Situation

Current local training/smoke set:

- Yes, we have the intended 30 scene-distinct YouTube clips.
- Train split: 20 clips from 20 different source videos.
- Test split: 10 clips from 10 different source videos.
- These are all YouTube-derived short source sections.
- They are uncalibrated monocular clips, so they are not GT novel-camera
  validation data.

Current local GT validation set:

- We have 8 paired source/target validation samples. This is enough for the
  first baseline validation loop.
- It is 4 scenes/source families with 2 held-out target camera angles each:
  AIST, Neural 3D Video, ViVo, and DeepView.
- Each sample is a 4-second synchronized source/target window.
- The loader samples 16 frames at 4fps and 128px from native local MP4s.
- Adding more GT scenes remains useful, but reaching 10 samples is no longer a
  blocker for the first pass.

Current camera-unification status:

- Not done yet. `load_multicam_val_sample` currently returns frames plus
  metadata, but not a renderer-ready `CameraSpec`.
- DeepView and ViVo have enough local calibration to build proper adapters.
- Neural 3D Video has `poses_bounds.npy`, but needs a careful LLFF-axis adapter.
- AIST now has fixed camera IDs, synchronized videos, and local AIST++ camera
  parameters; it still needs a canonical `CameraSpec` adapter before renderer
  code can treat those intrinsics/extrinsics as first-class cameras.

Renderer camera expectations and per-dataset adapter status are defined in:

```text
data/CAMERA_CONTRACT.md
```

## Current Local Smoke Data

The current local Mac default should stay small and fast:

- 30 scene-distinct source videos.
- 20 train clips and 10 test clips.
- One clip per source video.
- No train/test source-video overlap.
- 64px, 4fps, 16 frames for tiny local runs.
- A matched 256px, 4fps, 16-frame materialization exists for V-JEPA/local
  encoder comparisons on the same source-video split.

This is for fast baseline iteration on a Mac, not for final validation. It
checks that the trainer, dataloader, renderer path, and loss plumbing work on
different scenes without overfitting one camera rig or one semantic setting.

Baseline configs:

- `src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc`
- `src/train_configs/local_mac_scene_distinct_30_local_encoder_256_fast_mac_2048splats.jsonc`
- `src/train_configs/local_mac_scene_distinct_30_vjepa2_vitl_fpc16_256_frozen_256_fast_mac_2048splats.jsonc`

Run the 256px comparison with:

```bash
./src/dataset_scripts/youtube_scene_distinct_30_256_seed.sh build-clips
./src/train_scripts/train_local_mac_30_clip_vjepa2_256_baseline.sh both
```

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

## Scalable Pseudo-Multicam Practice Sources

There is a third category that should stay separate from both monocular
YouTube smoke clips and calibrated GT validation: scalable pseudo-multicam
practice data.

This category includes public monitor-wall videos, live-production multiview
recordings, broadcast/director-comms views, and sites such as BrettZone where
many events expose multiple synchronized camera feeds for one real-world
dynamic scene. These are valuable because they can cheaply provide many
multi-view-ish training examples:

```text
same event/time -> several camera feeds or tile crops -> cross-view pressure
```

They are not metric novel-view validation unless camera calibration and timing
are trustworthy. Use them for pretraining, robustness, implicit-camera
practice, camera-ID-conditioned training, tile-heldout diagnostics, and
qualitative cross-view checks. Keep held-out-camera PSNR/SSIM claims routed to
calibrated sources such as DeepView, ViVo, Neural 3D Video, AIST, or CMU
Panoptic/DomeDB.

Current routed examples:

- YouTube production multiview / director-comms playlists:
  `data/youtube_multiview_comms/candidates/`.
- NHRL BrettZone robot fights: many short, dynamic, multi-camera fight records
  with public metadata/API surfaces and per-camera MP4 URLs embedded in fight
  review pages.
- TVMCE / Virtual Film Studio multi-camera TV-show editing dataset: six
  synchronized tracks per scene plus edited-video supervision across concerts,
  sports, gala shows, and contests.

## CMU Panoptic / DomeDB Validation Candidate

CMU Panoptic Studio / DomeDB is a high-value calibrated dynamic multicam
candidate:

- Site: <http://domedb.perception.cs.cmu.edu/>
- The homepage describes 480 VGA camera views, 30+ HD views, 10 RGB-D sensors,
  hardware synchronization, and calibration.
- The homepage reports 65 sequences, 5.5 hours, and about 1.5 million 3D
  skeletons.
- It includes multi-person social interaction scenes with labels such as 3D
  body pose, 3D facial landmarks, transcripts, and speaker IDs.
- The license is research-only, non-commercial, and the dataset or modified
  versions cannot be redistributed without organizer permission.
- The site also links PtCloudDB from 10 Kinects with 41 RGB videos and 6+ hours
  of data.

Use this as an external curated dataset lane, not a scrape target. First pass
should inspect one sequence for camera-video accessibility, calibration file
shape, frame timing, and whether a 16-frame/4fps held-out-camera window can be
adapted into `data/CAMERA_CONTRACT.md`.

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

## DeepView Video Validation Track

DeepView Video is a strong GT multi-camera validation source:

- Repository: <https://github.com/augmentedperception/deepview_video_dataset>
- Paper site: <https://augmentedperception.github.io/deepviewvideo/>
- The README describes 15 raw multi-camera light-field video scenes.
- Each scene is a separate zip archive with up to 46 synchronized camera videos.
- Each scene includes `models.json` camera calibration from structure from
  motion.
- Cameras are Yi4k action-camera fisheye videos, so evaluation must preserve
  fisheye projection and radial distortion metadata.

Local intake:

- Config: `src/dataset_configs/deepview_video_seed.jsonc`.
- Script: `./src/dataset_scripts/deepview_video_seed.sh`.
- Local root: `data/external/deepview_video/`.
- Full configured archive set: about 65.5 GB compressed.
- Current downloaded scenes: `03_Dog` and `15_Branches`.

`03_Dog` is the first usable local validation scene: 41 calibrated fisheye
camera videos, 2560x1920, about 5.0 seconds at 29.97fps. `15_Branches` is useful
as a structure/calibration smoke scene, but it has only 10 frames per camera and
is too short for the current 16-frame/4fps validation window.

## Neural 3D Video Validation Expansion TODO

Neural 3D Video release v1.0 is another GT multi-camera source:

- Release: <https://github.com/facebookresearch/Neural_3D_Video/releases/tag/v1.0>
- Local seed config: `src/dataset_configs/neural_3d_video_seed.jsonc`.
- Current downloaded scene: `coffee_martini`.
- Each extracted scene provides synchronized camera MP4s plus
  `poses_bounds.npy`; camera normalization still needs an adapter.

Potential validation expansion scenes from the release:

- `cook_spinach.zip`, about 1.21 GB compressed.
- `cut_roasted_beef.zip`, about 1.14 GB compressed.
- `flame_salmon_1`, about 4.99 GB compressed across four split archive parts.
- `flame_steak.zip`, about 1.20 GB compressed.
- `sear_steak.zip`, about 1.19 GB compressed.

TODO:

- Download one additional non-food-prep motion scene first, likely
  `flame_steak` or `sear_steak`, and inspect camera count/duration.
- Parse `poses_bounds.npy` into the same canonical camera schema as DeepView.
- Add at least two held-out-camera pairs per usable Neural 3D Video scene.

## ViVo Validation Source

ViVo is a large RGB-D multi-view volumetric-video dataset:

- Project site: <https://vivo-bvicr.github.io/>
- Temporary dataset folder: <https://drive.google.com/drive/folders/1uG4JB2GDWrIRMqmbI6NCP2kA0jUDAbvp?usp=sharing>
- Access form: <https://forms.office.com/e/gtKpYriSMJ>
- Processing code: <https://github.com/azzarelli/ViVo-DataProcessing>

The project site describes 14 RGB/depth video-camera pairs per capture session,
per-frame intrinsics/extrinsics, generated point clouds, and generated 2-D
masks. Unlike DeepView or Neural 3D Video, ViVo is not a simple stable GitHub
release download. If the local raw/compact `athlete_rows` data is deleted, we
need the project form/email or temporary Drive folder again.

## Current Local Multi-Camera Val Seed

The first local paired validation set is:

```text
data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/
```

Build it with:

```bash
./src/dataset_scripts/multicam_val_v1_seed.sh
```

It is intentionally not AIST-only. Current pairs:

- AIST Dance DB refined 10Mbps, `gBR_sBM_d04_mBR0_ch01`, `c01 -> c05`.
- AIST Dance DB refined 10Mbps, `gBR_sBM_d04_mBR0_ch01`, `c01 -> c09`.
- Neural 3D Video `coffee_martini`, `cam00 -> cam10`.
- Neural 3D Video `coffee_martini`, `cam00 -> cam20`.
- ViVo `athlete_rows`, `000404613112 -> 000497113112`.
- ViVo `athlete_rows`, `000404613112 -> 000516213112`.
- DeepView Video `03_Dog`, `camera_0001 -> camera_0015`.
- DeepView Video `03_Dog`, `camera_0001 -> camera_0040`.

Each pair writes source frames, held-out target frames, and a side-by-side
preview:

```text
clips/<sample_id>/summary.json
previews/<sample_id>.mp4
manifest.jsonl
dataset.json
```

Raw source videos stay at native FPS/resolution. The manifest stores source and
target video paths plus the synchronized time window. The loader samples the
current metric tensor target, 128px, 4fps, 16 frames, at load time. Side-by-side
preview MP4s are rendered separately at 30fps so human review is smooth while
the metric tensors stay tiny. AIST raw source videos are downloaded from the
10Mbps refined CSV and stay local. ViVo pairs are aligned by capture timestamp
overlap because the local train/test camera MP4s do not start at the same
wall-clock time. DeepView pairs preserve `models.json` paths plus source/target
fisheye projection and radial-distortion metadata in each manifest record.

## Ex4DGS Pretrained Checkpoint Track

Ex4DGS pretrained assets are a separate validation/evaluation track from raw
multi-camera video clips:

- Release: <https://github.com/juno181/Ex4DGS/releases/tag/v0.1>
- Selected assets: `coffee_martini.zip`, `Birthday.zip`, `Fabien.zip`.
- Local root: `data/external/ex4dgs_pretrained/`.
- Config: `src/dataset_configs/ex4dgs_pretrained_val_seed.jsonc`.
- Script: `./src/dataset_scripts/ex4dgs_pretrained_val_seed.sh`.

These are pretrained model/checkpoint bundles, not source videos. Each selected
asset extracts to `cameras.json`, `mean_metrics.json`, `all_metrics.json`,
`input.ply`, and final-iteration static/dynamic PLY point clouds. Current local
inventory:

- `coffee_martini`: 5400 camera records, final iteration 40000.
- `Birthday`: 800 camera records, final iteration 40000.
- `Fabien`: 800 camera records, final iteration 30000.

Do not mix these into `multicam_val_v1` as if they were synchronized raw
source/target MP4s. They become useful once we add either:

- a checkpoint evaluator that can render Ex4DGS PLY/camera bundles directly, or
- the matching raw Technicolor / Neural 3D Video frames so their cameras can be
  evaluated against ground truth in the same metric harness.

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
