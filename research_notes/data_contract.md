# DynaWorld Data Contract

This is the repo contract for training data. Keep data loader changes, manifest
schema changes, and README claims aligned with this file.

## Current Loader Families

DynaWorld has two reusable data paths today:

| Path | Loader | Main use | Loss surface |
| --- | --- | --- | --- |
| Single-sequence manifest | `src/train/sequence_data.py` | Broad same-view pretraining from one camera/window at a time | input video -> rendered same camera |
| Multicam bundle | `src/train/multicam_video_data.py` | Calibrated train-camera input and heldout-camera supervision | input view(s) -> rendered novel view |

The single-sequence path is the scale path. It can stream many videos without
copying frames by loading `explicit_video_window` rows lazily from the source
video. It also supports prepared frame clips and camera-json sequences.

The multicam path is the novel-view path. It loads a bundle with conditioning
frames, train-camera frames/cameras, and optional heldout-camera frames/cameras.
It is the path to use when the metric is "does the same world token render a
camera angle that was not encoded?"

These two paths are compatible at the training-scheduler level, but they are
not one mixed trainer yet. The remaining bridge is a sampler/trainer that takes
both manifests and alternates same-view and novel-view losses in one run.

## Supported Training Tasks

### Same Angle

Use this for scale pretraining and for static/dynamic token pressure when only
one camera is available.

- Input: one video window or prepared clip from one camera.
- Target: the same camera and time window.
- Loader: `load_manifest_sequence` / `load_manifest_sequences`.
- Example config:
  `src/train_configs/local_mac_scale_static_dynamic_vjepa_1k_video_pretrain_F32_256_16f_8192splats.jsonc`.
- Current behavior: lazy 1k training rows are loaded from
  `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl`.

Same-angle loss is useful, but it does not prove world-token behavior by
itself. It proves reconstruction under the encoded view.

### Novel Angle

Use this for the actual world-token check: train camera(s) go in, heldout camera
frames score the rendered result.

- Input: calibrated train camera view(s) from the same scene/time window.
- Target: heldout calibrated camera view(s).
- Loader: `load_multicam_video_bundle`.
- Example config:
  `src/train_configs/local_mac_scale_static_dynamic_vjepa_multicam_train2_holdout1_F32_256_16f_8192splats.jsonc`.
- Current manifests:
  - `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl`
  - `data/multicam_val/clip_sets/multicam_val_v1_chunked_128_4fps_16f/manifest.jsonl`
  - `src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f_manifest.jsonl`

Novel-angle loss should be reported separately from same-angle loss. Do not
collapse the two into one baseline number without naming the mixture.

### Mixed Same-View + Novel-View Training

This is the intended next unification:

1. Sample a same-view batch from the broad single-sequence manifest.
2. Sample a novel-view batch from a multicam manifest.
3. Run the same model/head.
4. Backprop both `same_view_recon` and `heldout_view_recon`, with separate logs.

The data contract supports this, but the mixed sampler/trainer is not complete
yet. Until it exists, run the 1k same-view pretrain and multicam novel-view
smokes as separate lanes.

## Current 1k Same-View Train Rows

Generated artifact:
`data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl`.

Build config:
`src/dataset_configs/single_video_pretrain_1k_manifest.jsonc`.

Builder:
`src/dataset_scripts/build_single_video_pretrain_manifest.py`.

Current train split:

| Source label | Rows | Notes |
| --- | ---: | --- |
| `youtube_curated_spans_raw` | 260 | raw YouTube windows |
| `vivo_train_singleview` | 260 | multicam dataset loaded as single-view windows |
| `neural3d_coffee_singleview` | 193 | synthetic/dynamic NeRF video loaded as single-view windows |
| `deepview_singleview` | 153 | lightfield/multiview dataset loaded as single-view windows |
| `aist_train_singleview` | 51 | AIST dance cameras loaded as single-view windows |
| `youtube_scene_distinct_segments` | 26 | mined YouTube source segments |
| `youtube_scene_distinct_256_clip` | 16 | prepared frame clips |
| `youtube_curated_spans_64_clip` | 15 | prepared frame clips |
| `local_mac_30_synthetic_clip` | 13 | prepared local synthetic clips |
| `youtube_high_motion_smokes` | 12 | high-motion YouTube smoke windows |
| `blender_sintel_camera_json` | 1 | camera-json synthetic sequence |

By record type:

- `single_view_video_window`: 955 train rows
- `frame_clip_sequence`: 44 train rows
- `synthetic_camera_json_sequence`: 1 train row

The generated `eval_manifest.jsonl` is currently empty for this 1k artifact.
Use the heldout sources below for validation until the builder is extended to
emit a non-empty eval split from non-leaking sources.

## Current Heldout Pools

Single-camera heldouts:

| Manifest | Rows | Heldout role |
| --- | ---: | --- |
| `data/youtube_scene_distinct/clip_sets/youtube_scene_distinct_30_256_4fps_16f/manifest.jsonl` | 30 total, 10 test | same-view validation |
| `data/youtube_curated_spans/clip_sets/youtube_curated_spans_64_4fps_16f/manifest.jsonl` | 19 total, 1 test | same-view validation |
| `data/clip_sets/local_mac_30_64_4fps_16f/manifest.jsonl` | 30 total, 10 test | same-view validation |

Multicam heldouts:

| Manifest | Rows | Datasets |
| --- | ---: | --- |
| `data/multicam_val/clip_sets/multicam_val_v1_128_4fps_16f/manifest.jsonl` | 8 | AIST 2, DeepView 2, Neural3D 2, ViVo 2 |
| `data/multicam_val/clip_sets/multicam_val_v1_chunked_128_4fps_16f/manifest.jsonl` | 14 | AIST 4, DeepView 2, Neural3D 4, ViVo 4 |
| `src/dataset_configs/multicam_train2_holdout1_5sample_128_4fps_16f_manifest.jsonl` | 5 | smoke set with train and heldout camera lists |

The 1k builder excludes known heldout target paths from the broad same-view
train split. That exclusion is path/camera-level, not a semantic scene-level
proof. Treat it as a leakage guard, not as a substitute for a clean validation
protocol.

## Manifest Row Contracts

Single-sequence rows must identify how frames are loaded:

- `record_type`: `single_view_video_window`, `frame_clip_sequence`, or
  `synthetic_camera_json_sequence`.
- `frame_source`: `explicit_video_window`, `summary_sampled`, or `camera_json`.
- `frame_count`, `fps`, `target_size`, and `split`.
- `current_loader_compatible`: true when the active trainer path can read it.

For `explicit_video_window`, rows must include `video_path`, `start_seconds`,
and `duration_seconds`.

For `summary_sampled`, rows point back to a prepared clip/frames manifest that
`load_manifest_sequence` can read.

For `camera_json`, rows include a sequence directory with a camera JSON file.

Multicam rows must include source and target camera/video fields. The richer
train2-holdout1 records also include `train_cameras`, `heldout_cameras`,
`anchor_camera`, and `condition_camera`.

## Static and Dynamic Data

Do not separate static and dynamic data into different loader systems. The
loader contract is about available camera supervision:

- Single camera, static or dynamic: same-angle reconstruction.
- Multiple calibrated cameras, static or dynamic: same-angle plus novel-angle
  reconstruction.

The model may still use static and dynamic token groups internally. The data
loader should only expose frames, times, camera specs, split metadata, and
whether heldout camera supervision exists.

## Commands

Regenerate the broad no-copy 1k manifest:

```bash
PYTHONPATH=src/train uv run python src/dataset_scripts/build_single_video_pretrain_manifest.py
```

Run the current 1k same-view scale config:

```bash
PYTHONPATH=src/train uv run python src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_scale_static_dynamic_vjepa_1k_video_pretrain_F32_256_16f_8192splats.jsonc
```

Run a multicam novel-view lane with the current per-record config/script flow:

```bash
./src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh
```

Audit counts quickly:

```bash
jq -r '.source_label' data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl | sort | uniq -c
jq -r '.record_type' data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl | sort | uniq -c
```

## Next Code Unification

The next reusable abstraction is a mixed data scheduler, not another manifest
format. Same-view manifest loading and lazy/eager sampling now live in
`sequence_data.ManifestSequenceSampler`; it deliberately stays on the
single-sequence side of the contract and does not hide multicam semantics. The
mixed batch boundary lives in `src/train/mixed_data_scheduler.py`; it returns
typed same-view and novel-view batches with explicit loss names. The first
trainer consumer is
`src/train/train_mixed_same_heldout_implicit_dynamic.py`, dispatched by
`arch=mixed_same_heldout_precomputed_feature_implicit_camera`. The checked-in
smoke config
`src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`
proves the batch boundary executes alternating same-view and heldout-view
optimizer steps with separate logs; it is still only a smoke, not a baseline.
The mixed trainer now delegates the full schedule branch to
`mixed_data_scheduler.sample_mixed_step_batch(...)`, passing a lazy same-view
sequence provider so novel-only steps do not force same-view manifest loading.
The post-refactor checked-in smoke passed at
`wandb/offline-run-20260521_173114-em1oaiqp`.
After extracting `ManifestSequenceSampler`, the same checked-in smoke passed
again offline at `wandb/offline-run-20260521_171453-7wqptf1i`.
After extracting the F32 alpha/background guard into `RGBReconObjective`, the
same smoke passed again at `wandb/offline-run-20260521_171924-mkj9af97`; a
cheap single-cam F32 alpha-aware smoke also passed at
`wandb/offline-run-20260521_171908-pgv52pgm`.
After extracting the shared multicam train/heldout reconstruction loop into
`MulticamPrecomputedFeatureImplicitTrainer._recon_loss_for_views(...)`, the
same checked-in mixed smoke passed again at
`wandb/offline-run-20260521_172310-9iwq2eer`, with a fresh current-state rerun
at `wandb/offline-run-20260521_172710-2xs5airh`. The split between
`same_view_recon` and `heldout_view_recon` remains visible; the extraction only
removes duplicated per-view render/loss mechanics.
The rendered-view loss helper was then shared with the camera-swap path as
well; a 1-step oracle-relative camera-swap smoke passed at
`wandb/offline-run-20260521_173425-bf4yc6h0`, and the checked-in mixed smoke
passed again at `wandb/offline-run-20260521_173547-6qpl53pz`. This does not
change the data contract, but it means multicam train-view, heldout-view, and
camera-swap renders now enforce the same alpha/background and preview mechanics.
A good trainer-facing minimum API is:

```text
same_view_manifest: path
multicam_manifest: path
same_view_weight: float
novel_view_weight: float
```

It should return batches with an explicit `loss_kind`:

- `same_view`: one sequence, one camera path, same-view target.
- `novel_view`: multicam bundle, train cameras, heldout-camera target.

That keeps the core distinction visible in logs, baselines, and papers while
letting the model share the V-JEPA/static/dynamic-token/splat decoder path.
The remaining work is to promote this from smoke bridge to benchmark artifacts,
with both losses visible in W&B/result rows.
