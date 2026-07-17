# STAR UVT Timing Boundary Cleanup

Date: 2026-05-21 21:41:59 +07

## Goal

Continue trainer modularization by removing duplicated timing-summary logic
from STAR UVT train/probe scripts while preserving row schemas and timing keys.

## Change

- Added `src/train/star_uvt_timing.py`.
- Added `tests/test_star_uvt_timing.py`.
- Rewired:
  - `src/train/train_star_uvt_feature_overfit.py`
  - `src/train/train_star_uvt_feature_rgb_probe.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`

The new helper owns:

- `mean_timing_ms(...)`
- `timing_trace_summary_ms(...)`

It preserves the prior behavior:

- inferred-key mean timing on an empty list returns `{}`
- explicit-key mean timing on an empty list returns `0.0` for each requested key
- empty trace summaries keep the same key shape with `None` values

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_timing.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_star_uvt_timing.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_timing.py \
  tests/test_star_uvt_outputs.py \
  tests/test_star_uvt_models.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_render_modes.py \
  tests/test_star_uvt_colorizers.py \
  tests/test_star_uvt_checkpoints.py -q
```

Result: `23 passed in 0.89s`.

Runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py /tmp/star_uvt_outputs_smoke_2step.jsonc
```

Result: pass true, loss decreased `0.18602049723267555 -> 0.11917690932750702`,
zero tile overflow, and timing summary fields were emitted through the shared
helper.

## Remaining Cleanup

The broad modularization goal remains open. This slice only moved repeated
timing-summary mechanics out of three STAR UVT scripts.
