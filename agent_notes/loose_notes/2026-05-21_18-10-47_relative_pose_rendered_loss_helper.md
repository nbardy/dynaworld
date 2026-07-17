# Relative Pose Rendered Loss Helper

## Context

The multicam trainer already had
`MulticamPrecomputedFeatureImplicitTrainer._rendered_view_recon_loss(...)` for
the common mechanics around a rendered view:

- require alpha-aware output when F-channel random-background composition is
  active
- profile and compute the reconstruction loss
- retain one training preview render and optional feature-PCA tensor

`MulticamRelativePoseImplicitTrainer.camera_swap_recon_loss(...)` still carried
its own copy of those mechanics inside the learned-residual/oracle-relative
camera-swap loop. That made the relative-pose branch a drift risk for the same
alpha/background cheat guard we care about in feature splatting.

## Change

- Replaced the duplicated relative-pose camera-swap rendered-view loss block
  with a call to the inherited `_rendered_view_recon_loss(...)` helper.
- Kept relative-pose-specific math local: source grouping, residual camera
  transforms, cycle loss, bank-rate aggregation, and rendered-count handling
  remain in `MulticamRelativePoseImplicitTrainer`.

No camera math, renderer math, optimizer behavior, or convergence target
changed.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_multicam_relative_pose_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py
```

Result: passed.

Temporary learned-residual camera-swap smoke:

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
cfg['logging']['video_log_every'] = 1000
cfg['logging']['always_log_last_step'] = False
cfg['logging']['log_initial_media'] = False
cfg['logging']['wandb_run_name'] = 'relative-pose-rendered-view-helper-smoke'
run_training(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_180825-m2xarb7g`.

Focused relative-pose import/config suite:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_multicam_relative_pose_trainer.py -q
```

Result: `13 passed in 1.45s`.

## Interpretation

This check proves only that the refactored relative-pose camera-swap plumbing
launches and still hits the shared alpha/background/loss helper. It does not
prove the training math converges. The next research proof remains a real W&B
trace with media and loss curves.
