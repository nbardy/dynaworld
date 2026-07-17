# PowerFoam Direct Config Boundary

## Context

Continuation of the trainer modularization goal. `train_powerfoam_direct.py`
still carried its defaults and `resolve_config(...)` inline, unlike the Metal
and Dynamic PowerFoam trainers that now route config normalization through
small config modules.

This was a clean extraction because Direct config normalization is pure:
it only applies defaults, normalizes path values, clamps neighbor count, and
validates shape/count fields. Direct render/loss/training logic stayed local.

## Changes

- Added `src/train/powerfoam_direct_config.py`.
  - Owns `DATA_DEFAULTS`, `MODEL_DEFAULTS`, `RENDER_DEFAULTS`,
    `TRAIN_DEFAULTS`, `LOSS_DEFAULTS`, and `LOGGING_DEFAULTS`.
  - Owns `resolve_config(...)`.
- Updated `src/train/train_powerfoam_direct.py` to import and re-export the
  config names for compatibility.
- Removed config-only imports (`apply_defaults`, `resolved_config`) and the
  unused `json` import from the Direct trainer.
- Updated `CODE_ORGANIZATION.md` and
  `TODO/trainer_landscape_unification.md`.

## Validation

Commands run from repo root:

```bash
rtk .venv/bin/python -m py_compile src/train/powerfoam_direct_config.py src/train/train_powerfoam_direct.py
rtk env PYTHONPATH=src/train .venv/bin/python - <<'PY'
from pathlib import Path
from powerfoam_direct_config import resolve_config as module_resolve
from train_powerfoam_direct import LOSS_DEFAULTS, RENDER_DEFAULTS, resolve_config as trainer_resolve
cfg = {'data': {'video_path': 'test_data/test_video_small_128_4fps.mp4'}, 'model': {'cells': 2, 'neighbor_count': 99}, 'render': {'background': [0, 0, 0]}, 'train': {'steps': 1}, 'losses': {}, 'logging': {}}
a = module_resolve(cfg)
b = trainer_resolve(cfg)
print('direct_config_alias_ok', a == b, b['model']['neighbor_count'], isinstance(b['data']['video_path'], Path), LOSS_DEFAULTS['l1_weight'], RENDER_DEFAULTS['render_size'])
PY
rtk env PYTHONPATH=src/train:third_party/powerfoam-metal uv run --with pytest python -m pytest tests/test_powerfoam_direct.py -q
```

Results:

- `py_compile` passed.
- Config alias smoke printed `direct_config_alias_ok True 1 True 1.0 128`.
- Full `tests/test_powerfoam_direct.py`: `43 passed, 2 skipped in 7.69s`.
  The run emitted the known parent `pyproject.toml` warning but exited `0`.

## Next

Keep using the same pattern: pure config/default boundaries belong in small
modules; Direct-specific render/loss/artifact behavior should stay local until
there is a genuinely shared helper contract.
