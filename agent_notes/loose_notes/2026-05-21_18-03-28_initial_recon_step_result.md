# Initial Recon Step Result

## Context

After the shared run-loop extraction, the base implicit-camera trainer and
`KnownCameraTrainer` still duplicated the initial eval payload:

- render the first eval/train clip through `render_decoded_rgb_clip(...)`
- compute reconstruction loss
- compute optional V-JEPA feature loss
- keep the first preview render and optional feature-PCA tensor
- build the initial `StepResult`

The branches differ in how they prepare cameras and decode the clip, so those
parts remain branch-local. The render/loss/payload tail is the same shape.

## Change

- Added `Trainer.initial_recon_step_result(...)`.
- `Trainer.initial_step_result(...)` now uses the helper after implicit-camera
  decode and camera regularizer construction.
- `KnownCameraTrainer.initial_step_result(...)` now uses the helper after
  known-camera decode and bank-rate construction.

No training-step math, backward path, optimizer behavior, renderer path, or
validation-video path changed.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py
```

Result: passed.

Temporary known-camera runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py /tmp/dynaworld_known_camera_runloop_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_180254-vgovjn1b`.

Base token-GS runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_overfit_video_token_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_180306-qosbrfqm`.

## Interpretation

This is an initial-eval payload cleanup. It proves the shared helper works for
both the implicit-camera and known-camera branches, but it is still plumbing
evidence rather than convergence or quality evidence.
