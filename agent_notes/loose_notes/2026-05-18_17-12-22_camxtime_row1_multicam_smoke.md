# CamXTime row-one multicam smoke

## What changed

- Added CamXTime support to `src/train/multicam_video_data.py`.
- Added row-level train/heldout/anchor/condition camera split fallback so a manifest row can fully describe its multicam split.
- Added a tiny row-one manifest:
  - `src/dataset_configs/camxtime_row1_full_grid_smoke_manifest.jsonl`
  - train cameras: `camera_000`, `camera_020`
  - heldout camera: `camera_040`
  - anchor/condition camera: `camera_000`
- Added dataset metadata:
  - `src/dataset_configs/camxtime_row1_full_grid_smoke.jsonc`
- Added one-step train config:
  - `src/train_configs/local_mac_camxtime_row1_full_grid_multicam_smoke.jsonc`

## Local data slice

Source page: http://zheninghuang.github.io/camxtime_dataset

Dataset repo: https://huggingface.co/datasets/zhening/CamxTime

The committed manifest expects this ignored local subset:

```text
data/external/camxtime/extracted/CamxTime_eval/full_grid_renders/scene_1/camera_data.json
data/external/camxtime/extracted/CamxTime_eval/full_grid_renders/scene_1/camera_000.mp4
data/external/camxtime/extracted/CamxTime_eval/full_grid_renders/scene_1/camera_020.mp4
data/external/camxtime/extracted/CamxTime_eval/full_grid_renders/scene_1/camera_040.mp4
```

The MP4s are from the full-grid render archive, not the preprocessed `eval_gt`
trajectory videos. `camera_data.json` is the full-grid format with top-level
`intrinsics`, `n_cameras=120`, and per-camera `c2w`/`w2c` entries keyed by
integer camera index.

## Verification

Focused multicam tests:

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_multicam_video_data.py -q
```

Result: `9 passed in 1.11s`.

Real row-one loader smoke:

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -c "<load config, load_multicam_video_bundle, assert shapes/poses>"
```

Result:

```text
train_names ['camera_000', 'camera_020']
heldout_names ['camera_040']
frames (2, 4, 3, 64, 64) (1, 4, 3, 64, 64)
K0 [[64.6464614868164, 0.0, 32.0], [0.0, 64.6464614868164, 32.0], [0.0, 0.0, 1.0]]
w2c_translations [[9.5367431640625e-07, 9.5367431640625e-07, 2.384185791015625e-07], [0.2893104553222656, 0.42015838623046875, -0.03240084648132324]] [[0.6320972442626953, 0.7855443954467773, -0.12810277938842773]]
```

One-step train smoke:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train WANDB_MODE=offline \
  .venv/bin/python src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train_configs/local_mac_camxtime_row1_full_grid_multicam_smoke.jsonc
```

Result: completed on MPS with W&B offline run
`wandb/offline-run-20260518_171201-ud27obl0`. The run used
`video_encoder_backend=none`, `rig_init=camxtime`, train views
`camera_000/camera_020`, heldout `camera_040`, 4 frames, 64px input/render,
64 explicit Gaussians, and the dense renderer.

Initial and final smoke metrics were low as expected for an unconditioned
one-step shape/proof run, but the train loop exercised multi-angle loading,
external CamXTime poses, train-view render/loss, heldout render/eval, W&B media,
and optimizer stepping.

## Follow-up

- Promote this from a row-one smoke to a real CamXTime manifest builder if we
  want to sample many scenes/cameras from the 5GB full-grid archive.
- Add an eval-trajectory manifest later if we want `moving_forward`,
  `moving_backward`, `moving_zigzag`, `moving_bullettime`, or `moving_slowmo`
  videos as moving-camera heldout paths instead of fixed full-grid cameras.
