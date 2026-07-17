# STAR UVT Output Boundary Cleanup

Date: 2026-05-21 21:38:37 +07

## Goal

Continue the training-code modularization goal by removing another repeated
STAR UVT script pattern without changing trainer math or result schemas.

## Change

- Added `src/train/star_uvt_outputs.py`.
- Added `tests/test_star_uvt_outputs.py`.
- Rewired:
  - `src/train/train_star_uvt_feature_overfit.py`
  - `src/train/train_star_uvt_feature_rgb_probe.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`

The new helper owns:

- contact-sheet and side-by-side video writing
- side-by-side FPS fallback from output config to data FPS to `30.0`
- optional output-path normalization
- sorted pretty JSON row persistence
- stdout JSON row emission

The scripts still own their row schemas, W&B media keys, checkpoint policy, and
training/probe math.

## Related Documentation Cleanup

The same docs pass also recorded the previous `star_uvt_models.py` boundary:
feature-overfit, rendered-feature RGB probe, and feature-overfit
diagnostics/profilers now build `FeatureScreenTimeTubeModel` through
`build_feature_tube_model(...)` instead of constructing the prototype class
directly.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_outputs.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_star_uvt_outputs.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_outputs.py \
  tests/test_star_uvt_models.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_render_modes.py \
  tests/test_star_uvt_colorizers.py \
  tests/test_star_uvt_checkpoints.py -q
```

Result: `19 passed in 0.82s`.

Runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py /tmp/star_uvt_outputs_smoke_2step.jsonc
```

Result: pass true, loss decreased `0.18602049723267555 -> 0.11917690932750702`,
zero tile overflow, and shared output helper wrote:

- `/tmp/star_uvt_outputs_smoke_2step.json`
- `/tmp/star_uvt_outputs_smoke_contact.png`

## Remaining Cleanup

- More STAR UVT benchmark/prototype scripts still construct toy models and
  render configs directly. That is fine for pure synthetic fixtures, but
  config-based trainer/profiler paths should keep using `star_uvt_models.py`
  and `star_uvt_render_configs.py`.
- The broader modularization goal remains open. This slice only removed one
  repeated STAR UVT output boundary and documented the model-construction
  boundary.
