# STAR UVT Feature RGB Probe Trainer Module Split

## Context

`train_star_uvt_feature_rgb_probe.py` was still a registered probe implementation
with a CLI footer. Its config validation had already moved to
`star_uvt_feature_rgb_probe_config.py`, so the remaining file had a clean split:
target-grid RGB probe orchestration belongs in an owner module, while the
historical `train_*.py` path should stay as the CLI/backcompat entrypoint.

## Change

- Added `src/train/star_uvt_feature_rgb_probe_trainer.py`.
- Moved `run_probe(...)` there.
- Replaced `src/train/train_star_uvt_feature_rgb_probe.py` with a thin
  CLI/backcompat wrapper.
- Updated `trainer_registry.py` so `star_uvt_feature_rgb_probe` routes to
  `star_uvt_feature_rgb_probe_trainer`.
- Updated `tests/test_trainer_registry.py` to assert the owner-module route.

## Validation

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/star_uvt_feature_rgb_probe_trainer.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/trainer_registry.py \
  tests/test_trainer_registry.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_feature_rgb_probe.py tests/test_star_uvt_config_keys.py tests/test_trainer_registry.py -q
# 24 passed
```

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from star_uvt_feature_rgb_probe_trainer import run_probe as owner_run
from train_star_uvt_feature_rgb_probe import run_probe as wrapper_run
from trainer_registry import trainer_entry_for_arch
assert owner_run is wrapper_run
entry = trainer_entry_for_arch("star_uvt_feature_rgb_probe")
assert entry.module == "star_uvt_feature_rgb_probe_trainer"
assert entry.runner == "run_probe"
PY
```

```bash
PYTHONPATH=src/train WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  src/train/train.py /tmp/dynaworld_star_uvt_feature_rgb_probe_owner_split_smoke.jsonc
# passed; output: /tmp/dynaworld_star_uvt_feature_rgb_probe_owner_split_smoke_result.json
```

The runtime smoke was patched from
`src/train_configs/star_uvt_feature_targetgrid_vjepa_rgb_probe_hidden64_lr01_1000step.jsonc`
to use 4 frames, 64px, a `[2, 16, 16]` token grid, one training step, disabled
loss-decrease requirement, disabled W&B, null checkpoint/media outputs, and a
`/tmp` result JSON. The first attempt used `[4, 4, 4]`, which correctly failed
because the V-JEPA target produced 512 tokens; the corrected smoke then passed
through `src/train/train.py -> trainer_registry -> star_uvt_feature_rgb_probe_trainer`.

## Handoff

The rendered-feature RGB probe remains larger and should be split separately
because it owns native sparse-pixel VJP train paths and checkpoint resume logic.
The feature-overfit trainer remains the large STAR implementation owner.
