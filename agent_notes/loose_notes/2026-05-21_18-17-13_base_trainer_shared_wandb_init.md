# Base Trainer Shared W&B Init

## Context

`train_logging.init_wandb_run(cfg)` already owned W&B initialization for the
shared `logging.wandb_*` config contract, but the main
`Trainer` in `train_video_token_implicit_dynamic.py` still carried its own
`wandb.init(...)` kwargs block.

That left the largest trainer outside the shared logging boundary and made
`logging.wandb_enabled` semantics inconsistent with PowerFoam and STAR UVT
probe/overfit scripts.

## Change

- `Trainer.__init__` now calls `init_wandb_run(self.cfg)` and stores the result
  as `self.wandb_run`.
- `Trainer.resolve_config(...)` now normalizes missing
  `logging.wandb_enabled` to `true`, preserving legacy configs that did not
  declare the key.
- `Trainer.val_log(...)` returns immediately when W&B is disabled.
- `Trainer.run_training_loop(...)` only calls `wandb.finish()` when a run was
  initialized.
- `Trainer.training_complete_message(...)` reports a disabled-W&B completion
  message when appropriate.

No model math, renderer math, loss composition, optimizer behavior, or training
schedule changed.

## Validation

Syntax/import check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python -m py_compile \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_logging.py \
  src/train/train_precomputed_feature_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py
```

Result: passed.

Config-normalization check:

```bash
PYTHONPATH=src/train rtk .venv/bin/python - <<'PY'
from config_utils import load_config_file
from train_video_token_implicit_dynamic import Trainer

cfg = load_config_file('src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc')
resolved = Trainer.resolve_config(cfg)
assert resolved['logging']['wandb_enabled'] is True
cfg['logging']['wandb_enabled'] = False
resolved = Trainer.resolve_config(cfg)
assert resolved['logging']['wandb_enabled'] is False
print('wandb_enabled defaults/override ok')
PY
```

Result: passed.

Default W&B path smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python - <<'PY'
from config_utils import load_config_file
from train_video_token_implicit_dynamic import run_training

cfg = load_config_file('src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc')
cfg['train']['steps'] = 1
cfg['logging']['log_every'] = 1
cfg['logging']['image_log_every'] = 1000
cfg['logging']['video_log_every'] = 1000
cfg['logging']['always_log_last_step'] = False
cfg['logging']['log_initial_media'] = False
cfg['logging']['wandb_run_name'] = 'base-trainer-shared-wandb-init-smoke'
run_training(cfg)
PY
```

Result: passed on MPS, W&B offline dir
`wandb/offline-run-20260521_181535-x31edkh2`.

Disabled W&B path smoke:

```bash
PYTHONPATH=src/train rtk .venv/bin/python - <<'PY'
from config_utils import load_config_file
from train_video_token_implicit_dynamic import run_training

cfg = load_config_file('src/train_configs/local_mac_tiny_30_video_token_smoke.jsonc')
cfg['train']['steps'] = 1
cfg['logging']['wandb_enabled'] = False
cfg['logging']['log_every'] = 1
cfg['logging']['image_log_every'] = 1000
cfg['logging']['video_log_every'] = 1000
cfg['logging']['always_log_last_step'] = False
cfg['logging']['log_initial_media'] = False
cfg['logging']['wandb_run_name'] = 'base-trainer-disabled-wandb-message-smoke'
run_training(cfg)
PY
```

Result: passed on MPS with no W&B run and ended with:

```text
DynamicVideoTokenGSImplicitCamera training complete (W&B disabled).
```

## Interpretation

This is a logging-boundary cleanup. It proves the largest trainer now follows
the same W&B initialization interface as the newer trainer/probe scripts and
that the disabled-W&B path is coherent. It does not prove anything about
convergence or visual quality.
