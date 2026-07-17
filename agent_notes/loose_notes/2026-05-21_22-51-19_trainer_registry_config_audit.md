# Trainer Registry Config Audit

## Goal

Continue trainer modularization by turning entrypoint routing into an explicit
contract. Checked-in train configs should either route through
`src/train/train.py` or be documented as research-CLI configs with a clear
launcher.

## Findings

- Current `src/train/` no longer contains the old typo/image/dynamicTokenGS
  shim files referenced by earlier cleanup notes.
- `src/train_configs` contains 20 distinct arch values.
- Before this pass, `star_uvt_feature_rgb_probe` had checked-in configs but was
  missing from `trainer_registry.py`.
- The gauge-field material-surfel configs and static/free-dynamic 3DGS gauge
  baseline configs are real checked-in configs, but they are not train.py routes.
  Their launchers live under `research_experiments/gauge_fields/`.
- The earlier `wandb_media.py` split left
  `research_experiments/gauge_fields/train.py` importing media helpers from
  `train_logging`; that would break import after the split.

## Changed

- Added `star_uvt_feature_rgb_probe` to `TRAINER_BY_ARCH`, routed to
  `train_star_uvt_feature_rgb_probe.run_probe`.
- Added `ExternalTrainerEntry` and `EXTERNAL_TRAINER_BY_ARCH` to
  `trainer_registry.py` for:
  - `gauge_fields_material_surfel` -> `research_experiments/gauge_fields/train.py`
  - `splat_baseline_free_dynamic_3dgs` ->
    `research_experiments/gauge_fields/train_splat_baseline.py`
  - `splat_baseline_static_3dgs` ->
    `research_experiments/gauge_fields/train_splat_baseline.py`
- Updated train.py re-exports for the external registry metadata.
- Updated `research_experiments/gauge_fields/train.py` to import
  W&B media helpers from `wandb_media`.
- Added `tests/test_trainer_registry.py`:
  - all checked-in config arches are registered or explicitly external
  - `star_uvt_feature_rgb_probe` routes to `run_probe`
  - external gauge arch errors point at the research launcher
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/trainer_registry.py \
  src/train/train.py \
  research_experiments/gauge_fields/train.py \
  tests/test_trainer_registry.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_trainer_registry.py \
  tests/test_config_factory_helpers.py::test_train_router_accepts_star_uvt_video_overfit_config \
  -q
```

Passed: `4 passed in 4.75s`.

Audit command:

```bash
PYTHONPATH=src/train .venv/bin/python - <<'PY'
from trainer_registry import TRAINER_BY_ARCH, EXTERNAL_TRAINER_BY_ARCH
from pathlib import Path
import re
arches=set()
for p in Path('src/train_configs').glob('*.json*'):
    m=re.search(r'"arch"\s*:\s*"([^"]+)"', p.read_text())
    if m: arches.add(m.group(1).lower())
print('total_config_arches', len(arches))
print('registered', sorted(arches & set(TRAINER_BY_ARCH)))
print('external', sorted(arches & set(EXTERNAL_TRAINER_BY_ARCH)))
print('missing', sorted(arches - set(TRAINER_BY_ARCH) - set(EXTERNAL_TRAINER_BY_ARCH)))
PY
```

Result: `total_config_arches 20`, `missing []`.

## Remaining

The external gauge-field scripts are still argparse-first CLIs. Fold them into
`src/train/train.py` only if we decide to refactor their giant `main()` bodies
into `run_training(config)` functions; do not fake that through a wrapper that
silently drops CLI semantics.
