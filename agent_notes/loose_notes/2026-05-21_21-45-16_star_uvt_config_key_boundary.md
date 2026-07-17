# STAR UVT Config-Key Boundary Cleanup

Date: 2026-05-21 21:45:16 +07

## Goal

Continue trainer modularization by removing repeated STAR UVT config-section
key contracts without changing branch-specific config semantics.

## Change

- Added `src/train/star_uvt_config_keys.py`.
- Added `tests/test_star_uvt_config_keys.py`.
- Rewired:
  - `src/train/star_uvt_feature_config.py`
  - `src/train/train_star_uvt_feature_rgb_probe.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`
  - `src/train/train_star_uvt_video_overfit.py`

The new helper owns shared required-key tuples and validation helpers for:

- common STAR UVT `data`
- common STAR UVT `colorize`
- common STAR UVT `output`
- output configs that also require `checkpoint`
- common STAR UVT `logging`

Branch-specific `train`, `probe`, `feature_uvt`, `uvt`, and `per_frame` keys
remain local to the relevant scripts/config modules.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_config_keys.py \
  src/train/star_uvt_feature_config.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  src/train/train_star_uvt_video_overfit.py \
  tests/test_star_uvt_config_keys.py \
  tests/test_star_uvt_feature_rgb_probe.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_config_keys.py \
  tests/test_star_uvt_feature_rgb_probe.py \
  tests/test_star_uvt_timing.py \
  tests/test_star_uvt_outputs.py \
  tests/test_star_uvt_models.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_render_modes.py \
  tests/test_star_uvt_colorizers.py \
  tests/test_star_uvt_checkpoints.py -q
```

Result: `34 passed in 2.35s`.

Runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py /tmp/star_uvt_outputs_smoke_2step.jsonc
```

Result: pass true, loss decreased `0.18602049723267555 -> 0.11917690932750702`,
zero tile overflow.

## Remaining Cleanup

The broad modularization goal remains open. This slice only centralizes common
STAR UVT config-section key validation; branch-specific config logic remains
local by design.
