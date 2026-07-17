# Known Camera Render Full Inheritance

## Context

`KnownCameraTrainer.render_full_sequence(...)` was a duplicate of the base
`Trainer.render_full_sequence(...)` implementation:

```python
return _render_full_sequence_impl(
    self.cfg, self.model, sequence_data, self._eval_decode_clip, self._eval_render_clip
)
```

The base method already dispatches through virtual methods. Since
`KnownCameraTrainer` overrides `_eval_decode_clip(...)` to pass known cameras,
the full-sequence wrapper itself does not need to be duplicated.

## Change

- Removed the redundant `KnownCameraTrainer.render_full_sequence(...)` override.
- Known-camera validation rendering now inherits the base implementation and
  still uses `KnownCameraTrainer._eval_decode_clip(...)` for camera ownership.

No rendering math, decode math, validation media payload, or training behavior
changed.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py
```

Result: passed.

Known-camera inherited full-sequence validation smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python - <<'PY'
from config_utils import load_config_file
from train_video_token_implicit_dynamic import run_training

cfg = load_config_file('/tmp/dynaworld_known_camera_runloop_smoke.jsonc')
cfg['logging']['video_log_every'] = 1
cfg['logging']['image_log_every'] = 1000
cfg['logging']['log_initial_media'] = False
cfg['logging']['wandb_run_name'] = 'known-camera-inherited-render-full-smoke'
run_training(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_180535-tcwvntcd`.

## Interpretation

This is a small deletion, but it removes a redundant override from a trainer
branch and proves the inherited base validation path still works for known
cameras.
