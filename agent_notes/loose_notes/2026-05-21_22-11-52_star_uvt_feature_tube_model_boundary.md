# STAR UVT Feature Tube Model Boundary

## Goal

Continue the modularization goal by removing an inverted dependency: active
`src/train` STAR UVT modules were still importing feature-tube model/config
objects from `research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py`.

## What Changed

- Added `src/train/star_uvt_feature_tube_model.py` as the shared owner for:
  - `FeatureTubeRenderConfig`
  - `FeatureScreenTimeTubeModel`
  - `make_uvt_grid(...)`
  - `dense_render_feature_tubes(...)`
  - `render_model_features(...)`
  - `make_default_colorizer(...)`
  - `colorize_and_compose(...)`
- `src/train/star_uvt_models.py` now imports `FeatureScreenTimeTubeModel` from
  the new train module.
- `src/train/star_uvt_render_configs.py` now imports `FeatureTubeRenderConfig`
  from the new train module.
- `research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py`
  now re-exports the shared implementation and keeps the old gate/smoke runner
  plus old import names for benchmark compatibility.
- Tests that need the model/config contract now import from
  `star_uvt_feature_tube_model` instead of the research prototype.

## Why

The model/config contract is no longer just a one-off prototype. It is used by
the first-class STAR UVT feature trainer, render-config factory, model factory,
support diagnostics, and feature-target tests. Keeping it in
`research_experiments/` made active train code depend on a historical benchmark
script. This slice moves the shared interface into `src/train` while preserving
old script imports through the prototype shim.

## Validation

- Import audit:

```text
rg -n "research_experiments\.star_uvt_feature_tubes\.dense_feature_tube_prototype|from dense_feature_tube_prototype import" src/train tests -g '*.py'
```

Result: no matches.

- `py_compile` passed for the new module, model/render-config factories, dense
  prototype shim, support prototypes, and focused tests.
- Focused tests passed:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_models.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_visibility_support_bridge.py -q
```

Result: `40 passed in 1.27s`.

- Dense prototype smoke still passes through the compatibility shim:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py --smoke
```

Result: loss decreases and geometry/feature gradients are seen.

- Runtime STAR trainer smoke still passes through `build_feature_tube_model(...)`:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_8f_64_directatomic_20step.jsonc
```

Result: pass true, loss decreases `0.1860204972 -> 0.0416746489`, final full
RGB PSNR `13.9736485`, zero tile overflow, and model/colorizer gradients are
seen.

## Current State

This closes the active train-code dependency on the dense research prototype.
The prototype is still useful as a small gate runner and compatibility surface
for old benchmarks, but it is no longer the owner of the reusable model/config
contract.

Next cleanup candidates:

- Render-dispatch convergence: keep renderer-specific mechanics local, but make
  render-mode selection and payload contracts easier to audit.
- Mixed same-view plus heldout scheduler evidence: run a real W&B-enabled
  benchmark row before claiming the bridge is more than plumbing.
- Alpha-background one-off script retirement: only delete old runners after
  confirming the configurable ablation runner covers the old parameter surface.
