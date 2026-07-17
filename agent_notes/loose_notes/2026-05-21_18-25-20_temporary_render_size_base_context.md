# Temporary Render Size Base Context

## Context

`MulticamRelativePoseImplicitTrainer` carried its own
`temporary_render_size(...)` context manager and a local `_dense_grid_for_render_size(...)`
helper. The generic mechanics were not relative-pose-specific:

- save current render size, dense grid, renderer mode, and effective Gaussian
  count
- activate a temporary render size
- restore the previous state in `finally`
- reuse cached dense grids by render size

The only relative-pose-specific part is `_activate_render_size(...)`, because it
also needs token-detail-aware renderer dispatch.

## Change

- Added `Trainer.temporary_render_size(...)` to the base token-GS trainer.
- Removed the duplicate relative-pose `temporary_render_size(...)` context.
- Removed the duplicate relative-pose `_dense_grid_for_render_size(...)` helper;
  the inherited base dense-grid cache is now used.
- Kept `MulticamRelativePoseImplicitTrainer._activate_render_size(...)` local so
  token-detail-aware renderer mode/effective Gaussian selection remains
  branch-specific.

No render math, loss math, camera math, optimizer behavior, or logging cadence
changed.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py
```

Result: passed.

Relative-pose inherited render-size context smoke:

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
cfg['logging']['wandb_run_name'] = 'relative-pose-inherited-temp-render-size-smoke'
run_training(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_182501-qqrr5zvd`.

## Interpretation

This is a render-dispatch plumbing cleanup. It proves the relative-pose path can
use the inherited render-size context while preserving its branch-specific
renderer activation. It does not prove model convergence or visual quality.
