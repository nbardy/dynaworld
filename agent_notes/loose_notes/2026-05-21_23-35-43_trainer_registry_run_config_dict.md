# Trainer Registry In-Memory Dispatch

## Context

The previous registry cleanup gave diagnostics an arch-aware config resolver,
but two STAR feature-tube experiment scripts still imported
`train_star_uvt_feature_overfit.run_training` directly. Those scripts patch
configs in memory before launching; they needed a config-dict runner rather than
the existing path-only `run_config(...)`.

## Change

- Added `run_config_dict(config, config_path="<config>")` to
  `src/train/trainer_registry.py`.
- Changed `run_config(path)` to delegate through `run_config_dict(...)`.
- Added a registry test that dispatches a fake in-memory config to a fake
  registered runner and returns its value.
- Routed these STAR feature-tube scripts through the registry:
  - `research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py`
  - `research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py`

## Result

After this pass, the only remaining direct imports found by:

```bash
rg -n "from train_star_uvt_feature_overfit import|run_star_uvt_training|from train_video_token_implicit_dynamic import" \
  src/train research_experiments tests
```

are structural:

- `train_precomputed_feature_implicit_dynamic.py` subclasses the base trainer.
- `tests/test_temporal_sampling.py` intentionally tests `Trainer`.

## Validation

Validation passed:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/trainer_registry.py \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py \
  research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py \
  tests/test_trainer_registry.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_trainer_registry.py \
  tests/test_config_factory_helpers.py \
  tests/test_train_cli.py -q
```

The focused pytest slice passed with 26 tests. The import scan only found the
expected structural base-trainer subclass/test imports.
