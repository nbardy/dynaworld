# Trainer Run Loop Unification

## Context

The base implicit-camera trainer and `KnownCameraTrainer` still had separate
`run(...)` implementations after the earlier payload and media cleanups. The
known-camera loop copied the same step-0 diagnostic, progress bar, per-step
render-size schedule, preview retention, `val_log(...)`, and `wandb.finish()`
shape.

That copied loop was already drifting: the base loop had profile-print hooks,
browser-export handling, and sequence-prefetch cleanup; the known-camera loop
had its own copy and did not call the common cleanup path.

## Change

- Added `Trainer.training_start_message(...)`.
- Added `Trainer.training_camera_message(...)`.
- Added `Trainer.training_complete_message(...)`.
- Added `Trainer.should_export_browser_after_training(...)`.
- Added `Trainer.print_training_header(...)`.
- Added `Trainer.run_training_loop(...)`.
- Replaced `KnownCameraTrainer.run(...)` with small hook overrides:
  - known-camera start banner
  - known/precomputed camera summary
  - known-camera completion message
  - no browser export after training

The trainer-specific math and backward paths are unchanged. The shared loop now
owns only the train-loop plumbing: step-0 diagnostic, progress bar, per-step
schedule application, preview-retention decision, optional profile-print hook,
`val_log(...)`, optional export, sequence-prefetch cleanup, and W&B finish.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py
```

Result: passed.

Temporary known-camera run-loop smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py /tmp/dynaworld_known_camera_runloop_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_175918-9acf7f5a`.

Focused scheduler/logging checks:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_temporal_sampling.py \
  tests/test_train_logging.py -q
```

Result: `15 passed in 1.28s`.

Base token-GS runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_overfit_video_token_smoke.jsonc
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_175933-cv0vixo9`.

## Interpretation

This is train-loop plumbing evidence. It confirms the base and known-camera
branches run through the shared loop without changing the math. It is not
evidence of convergence or baseline quality.
