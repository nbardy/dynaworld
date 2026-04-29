# Multicam two-heldout overlap setup

## Context

After the first `camera_0040` heldout run, the heldout looked too far outside
the train-pair overlap. The active train pair was:

- train: `camera_0001`, `camera_0015`
- original heldout: `camera_0040`

Geometry check on DeepView `03_Dog` showed `camera_0040` is extreme:

- `0001 -> 0040`: 71.3 degrees
- `0015 -> 0040`: 87.7 degrees

Better candidates:

- `camera_0013`: balanced interpolation-style heldout, 32.8 degrees from
  `0001` and 38.0 degrees from `0015`.
- `camera_0029`: near-outside heldout, 53.9 degrees from `0001` but only
  25.2 degrees from `0015`.

## Code change

Extended the multicam V-JEPA trainer from one heldout camera to a list of
heldout cameras:

- `data.multicam_heldout_cameras` is now supported.
- Single-camera `data.multicam_heldout_camera` remains backward compatible.
- `MulticamVideoBundle.heldout_frames` is now `[H, T, C, h, w]`.
- `LearnableCameraRig` stores heldout camera views as `[H, T, ...]`.
- Validation logs separate metrics/videos under keys like
  `Heldout0_camera_0013/Eval/PSNR` and
  `Heldout1_camera_0029/Eval/PSNR`.

## Config

Added:

```text
src/train_configs/local_mac_multicam_deepview_4cam_train2_holdout2_overlap_static_dynamic_96_32_precomputed_vjepa2_1_vitb_384_128_16f_8192splats.jsonc
```

It uses:

- train: `camera_0001`, `camera_0015`
- heldout: `camera_0013`, `camera_0029`
- condition: `camera_0001`
- anchor: `camera_0001`

The feature cache key intentionally remains the previous condition-camera key,
because the V-JEPA input video and sampled frame window are unchanged.

## Verification

Compiled:

```bash
rtk uv run python -m py_compile src/train/multicam_video_data.py src/train/camera_rig.py src/train/train_multicam_precomputed_feature_implicit_dynamic.py
```

Config normalized correctly:

```text
train cameras: ['camera_0001', 'camera_0015']
heldout cameras: ['camera_0013', 'camera_0029']
sample id: deepview_03_Dog_camera_0001_to_camera_0015
feature cache key: multicam-deepview-3cam-train2-test1-static-dynamic-96-32-vjepa2-1-vitb-384-128-16f-v1
```

CPU bundle/rig check passed:

```text
train_frames (2, 16, 3, 128, 128) ['camera_0001', 'camera_0015']
heldout_frames (2, 16, 3, 128, 128) ['camera_0013', 'camera_0029']
rig heldouts 2
```

## Baseline Reference

The old `camera_0040` 5k run completed at:

```text
https://wandb.ai/nbardy/dynaworld/runs/mc5k0427c
```

Final printed eval:

| View | SSIM | PSNR |
| --- | ---: | ---: |
| TrainView0 | 0.9549 | 30.7235 |
| TrainView1 | 0.9422 | 30.0817 |
| Heldout `camera_0040` | 0.1310 | 12.4675 |
