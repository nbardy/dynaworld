# STAR UVT RGB Video Trainer Module Split

## Context

`train_star_uvt_video_overfit.py` was a small but real registered trainer: it
resolved RGB STAR UVT video configs, called the external
`run_video_fit_comparison(...)` bridge, asserted optional loss decrease, logged
W&B media/rows, and wrote the result JSON. The config validation had already
been moved to `star_uvt_video_overfit_config.py`, so the remaining file had a
clear owner-module boundary plus a CLI footer.

## Change

- Added `src/train/star_uvt_video_trainer.py`.
- Moved `run_training(...)` and the local loss-decrease assertion there.
- Replaced `src/train/train_star_uvt_video_overfit.py` with a thin
  CLI/backcompat wrapper.
- Updated `trainer_registry.py` so `star_uvt_video_overfit` routes to
  `star_uvt_video_trainer`.
- Updated `tests/test_config_factory_helpers.py` to assert the owner-module
  route.

## Validation

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  src/train/star_uvt_video_trainer.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/trainer_registry.py \
  tests/test_config_factory_helpers.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_config_factory_helpers.py tests/test_star_uvt_config_keys.py tests/test_trainer_registry.py -q
# 29 passed
```

```bash
PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python - <<'PY'
from star_uvt_video_trainer import run_training as owner_run
from train_star_uvt_video_overfit import run_training as wrapper_run
from trainer_registry import trainer_entry_for_arch
assert owner_run is wrapper_run
entry = trainer_entry_for_arch("star_uvt_video_overfit")
assert entry.module == "star_uvt_video_trainer"
assert entry.runner == "run_training"
PY
```

```bash
PYTHONPATH=src/train WANDB_MODE=offline PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  src/train/train.py /tmp/dynaworld_star_uvt_video_owner_split_smoke.jsonc
# passed; output: /tmp/dynaworld_star_uvt_video_owner_split_smoke_result.json
```

The runtime smoke was patched from
`src/train_configs/star_uvt_rgb_testvideo_64f_512_directatomic_8192t_20step_media.jsonc`
to use 4 frames, 64px, 128 tubes, one training step, one render-benchmark
repeat, disabled loss-decrease requirement, disabled W&B, and `/tmp` outputs.
It exercised `src/train/train.py -> trainer_registry -> star_uvt_video_trainer`
and the direct-atomic Metal path without writing benchmark/media artifacts into
the repo.

## Handoff

This split is only the RGB STAR video overfit wrapper. The larger
`train_star_uvt_feature_overfit.py` remains an implementation owner and should
be approached through smaller helper/owner boundaries or a dedicated split with
the STAR feature tests and a runtime smoke.
