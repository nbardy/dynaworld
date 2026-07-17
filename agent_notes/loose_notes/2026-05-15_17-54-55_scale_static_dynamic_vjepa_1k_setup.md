# Scale Static/Dynamic V-JEPA 1k Setup

## What changed

- Added a lazy train-manifest load mode to `src/train/train_video_token_implicit_dynamic.py`.
  - `data.train_manifest_load_mode: "eager"` preserves the old behavior.
  - `data.train_manifest_load_mode: "lazy"` loads manifest metadata up front, loads one sequence for initialization, then samples a manifest row from disk per train step.
  - Scalar logging and startup messages now report the full lazy manifest count instead of only the one initialized sequence.
- Added a precomputed-feature safety guard in `src/train/train_precomputed_feature_implicit_dynamic.py`.
  - Lazy precomputed-feature training now rejects `features.release_extractor_after_prebake=true`.
  - Without that guard, V-JEPA would be released after prebaking only the first initialized clip, then uncached later samples would fail.
- Added a broad same-video scale-pretrain config:
  - `src/train_configs/local_mac_scale_static_dynamic_vjepa_1k_video_pretrain_F32_256_16f_8192splats.jsonc`
  - This uses V-JEPA2.1 ViT-B/384 features, F32 feature splatting, 96 static + 32 dynamic tokens plus the current token layout, 256px render/loss, 16 frames, 8192 splats, lazy manifest loading, and `keep_in_memory=false`.
- Worker B also added no-copy `explicit_video_window` loading in `src/train/sequence_data.py`, wired the frame-source literal in `src/train/runtime_types.py`, and covered it in `tests/test_sequence_data_single_frame.py`.
- Added build/train launchers:
  - `src/train_scripts/build_dynaworld_scale_1k_clip_dataset.sh`
  - `src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh`
- Worker A added the multicam train2/holdout1 per-record launcher/config:
  - `src/train_configs/local_mac_scale_static_dynamic_vjepa_multicam_train2_holdout1_F32_256_16f_8192splats.jsonc`
  - `src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh`
- Worker B added a deterministic manifest inventory path:
  - `src/dataset_configs/single_video_pretrain_1k_manifest.jsonc`
  - `src/dataset_scripts/build_single_video_pretrain_manifest.py`

## Data inventory

The loader-compatible clip builder scans:

- `data/youtube_scene_distinct/raw`
- `data/youtube_scene_distinct/segments`
- `data/youtube_curated_spans/raw`
- `data/youtube_curated_spans/high_motion_smokes`
- `data/external/aist_dance_db/raw/refined_10M_sBM`
- `data/external/neural_3d_video/extracted`
- `data/external/vivo/rgb_mp4/athlete_rows/train`
- `data/external/vivo/rgb_mp4/athlete_rows/test`
- `data/external/deepview_video/extracted`
- `data/blender_synthetic`

Dry-run counts:

- 1.0s stride, no per-source cap: 938 usable 16f/4fps/256px clips across 189 source videos.
- 0.5s stride, no per-source cap: 1000 planned 16f/4fps/256px clips across 190 source videos.
- `data/blender_synthetic` currently contributes only skipped unusable/too-short `.avi` files in this scan; it is included so materialized future renders enter the source pool automatically.

The primary no-copy manifest artifact is already generated at:

- `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl`
- `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/manifest.jsonl`
- `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/dataset.json`

Final no-copy artifact counts from Worker B:

- `train_manifest.jsonl`: 1000 rows
- `manifest.jsonl`: 1002 rows
- `heldout_manifest.jsonl`: 2 rows
- `eval_manifest.jsonl`: 0 rows
- unique train source paths: 153
- current-loader-compatible rows after the `explicit_video_window` loader patch: 1002/1002

Build command:

```bash
PYTHONPATH=src/train uv run python src/dataset_scripts/build_single_video_pretrain_manifest.py
```

Training command after the dataset exists:

```bash
./src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh check
./src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh train
```

## Validation run

- `bash -n` passed for:
  - `src/train_scripts/build_dynaworld_scale_1k_clip_dataset.sh`
  - `src/train_scripts/train_scale_static_dynamic_vjepa_1k_video_pretrain.sh`
  - `src/train_scripts/train_scale_static_dynamic_vjepa_multicam.sh`
- `py_compile` passed for:
  - `src/train/train_video_token_implicit_dynamic.py`
  - `src/train/train_precomputed_feature_implicit_dynamic.py`
  - `src/dataset_scripts/build_single_video_pretrain_manifest.py`
- Scale config check passed:
  - `arch=precomputed_feature_implicit_camera`
  - manifest path `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl`
  - `load_mode=lazy`
  - `features.keep_in_memory=false`
  - `features.release_extractor_after_prebake=false`
  - manifest count: 1000 train rows.
- Multicam config/launcher check passed on the checked-in 5-sample train2/holdout1 manifest.
- `build_single_video_pretrain_manifest.py --dry-run --limit 20 --no-probe-videos` passed and reported 22 current-loader-compatible records from already materialized frame/camera manifests.
- Lazy loader runtime smoke passed with `/tmp/dynaworld_lazy_tiny_smoke.json`.
  - It ran `WANDB_MODE=offline` for one step on the existing tiny 64px manifest.
  - It loaded 20 train manifest entries lazily and sampled from disk during training.
  - It was slow on MPS/dense renderer because step 0 still encoded validation videos; the smoke completed successfully.
- A real `explicit_video_window` loader smoke loaded the first 1k-manifest MP4 window into a `[16, 3, 64, 64]` CPU tensor.
- `tests/test_sequence_data_single_frame.py` passed with `uv run --with pytest python -m pytest ... -q`.
- Targeted `git diff --check` passed for the trainer/data-loader/builder/script files touched in this setup.

## Caveats

- The 1k pretrain lane is same-video reconstruction scale training. It should help amortize static/dynamic token learning across more clips, but it is not a novel-view benchmark.
- Novel-view train-in/heldout-out validation remains the multicam lane. The latest relative-pose V-JEPA config is still the stronger heldout-camera path, but it is query-conditioned: heldout/query camera V-JEPA features enter the relpose head, so do not describe it as pure source-only novel-view synthesis.
- Current YouTube clips are monocular. They can scale the static/dynamic representation and renderer training, but they do not by themselves provide calibrated heldout-camera loss.
- Do not claim a new baseline until a real run is launched, W&B logs, and `BASELINES.md` gets an appended dated row.
