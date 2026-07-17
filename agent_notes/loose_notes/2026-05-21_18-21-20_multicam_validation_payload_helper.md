# Multicam Validation Payload Helper

## Context

Base multicam and relative-pose trainers had the same validation-media assembly
after their render-generation branches:

- resize train-view targets to the active render size
- resize heldout targets when available
- call `multicam_validation_video_payload(...)`
- update `self.gt_video_logged`
- pass camera-rig metrics and fps
- update best-heldout PSNR/SSIM bookkeeping

The only real branch-specific part is how each trainer obtains
`train_rendered`, `heldout_rendered`, and `decoded_metrics`.

## Change

- Added
  `MulticamPrecomputedFeatureImplicitTrainer.multicam_validation_payload_from_renders(...)`.
- Base multicam `validation_video_payload(...)` now chooses external versus
  oracle-relative renders and then delegates payload assembly to the helper.
- `MulticamRelativePoseImplicitTrainer.validation_video_payload(...)` preserves
  its base-render-size guard and predicted-relative render path, then delegates
  the shared payload assembly to the inherited helper.
- Removed now-unused `multicam_validation_video_payload` and `resize_images`
  imports from the relative-pose trainer.

No render math, camera math, loss math, optimizer behavior, or logging cadence
changed.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py
```

Result: passed.

Base multicam video-path smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python - <<'PY'
from config_utils import load_config_file
from train_multicam_precomputed_feature_implicit_dynamic import run_training

cfg = load_config_file(
    'src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc'
)
cfg['train']['steps'] = 1
cfg['logging']['log_every'] = 1
cfg['logging']['image_log_every'] = 1000
cfg['logging']['video_log_every'] = 1
cfg['logging']['always_log_last_step'] = False
cfg['logging']['log_initial_media'] = False
cfg['logging']['wandb_run_name'] = 'multicam-shared-validation-payload-smoke'
run_training(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_182035-qluznetc`.

Relative-pose video-path smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python - <<'PY'
from config_utils import load_config_file
from train_multicam_relative_pose_implicit_dynamic import run_training

cfg = load_config_file(
    'src/train_configs/local_mac_multicam_deepview_3cam_train2_test1_rgb_pyramid_static_dynamic_smoke_32_2f_64splats.jsonc'
)
cfg['arch'] = 'multicam_relative_pose_implicit_camera'
cfg['train']['steps'] = 1
cfg['train']['camera_swap_mode'] = 'learned_residual'
cfg['train']['camera_swap_pairs_per_step'] = 1
cfg['train']['camera_swap_include_self'] = True
cfg['train']['camera_swap_include_cross'] = True
cfg['train']['relpose_feature_frame_mode'] = 'clip'
cfg['logging']['log_every'] = 1
cfg['logging']['image_log_every'] = 1000
cfg['logging']['video_log_every'] = 1
cfg['logging']['always_log_last_step'] = False
cfg['logging']['log_initial_media'] = False
cfg['logging']['wandb_run_name'] = 'relative-pose-shared-validation-payload-smoke'
run_training(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_182058-pwtxd0j6`.

## Interpretation

This is a validation-media boundary cleanup. It proves the base multicam and
relative-pose validation-video paths still execute and now share target
assembly, W&B media payload construction, rig metrics, fps, and best-heldout
bookkeeping. It does not prove model convergence or visual quality.
