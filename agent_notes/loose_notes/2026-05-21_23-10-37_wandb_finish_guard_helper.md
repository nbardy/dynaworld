# W&B Finish Guard Helper

## Context

After the CLI helper cleanup, the remaining live STAR UVT train/probe scripts
still imported `wandb` only for the repeated cleanup idiom:

```python
finally:
    if run is not None:
        wandb.finish()
```

The shared W&B run creation already lives in `train_logging.init_wandb_run(...)`,
so the matching finish guard belonged next to it. A follow-up check found the
same lifecycle shape in the Token-GS and PowerFoam-family trainers, using the
run object's `.finish()` method instead of global `wandb.finish()`.

## Change

- Added `train_logging.finish_wandb_run(run)`.
- The helper calls `run.finish()` when the run object exposes it and falls back
  to `wandb.finish()` for callers that only need global cleanup.
- Updated STAR UVT RGB overfit, feature overfit, feature RGB probe, and
  rendered-feature RGB probe to call the shared finish helper.
- Updated Token-GS, PowerFoam Direct, Dynamic Gauge Foam, Dynamic PowerFoam
  Metal, and PowerFoam Metal to call the shared finish helper.
- Removed local `wandb` imports from those STAR UVT train/probe modules.
- Added focused `tests/test_train_logging.py` coverage for `None`, run-object
  finish, and global-fallback finish behavior.

This is intentionally smaller than a W&B context manager. The train/probe loops
still differ in meaningful ways, but final run cleanup is now one shared
boundary.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_logging.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_powerfoam_metal.py \
  tests/test_train_logging.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py tests/test_train_cli.py -q
```

After extending the helper to Token-GS and PowerFoam-family trainers, the
focused validation was rerun:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_logging.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_powerfoam_metal.py \
  src/train/train_star_uvt_video_overfit.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_train_logging.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_logging.py tests/test_train_cli.py tests/test_train_optim.py -q
```

Result: `16 passed in 5.50s`.
