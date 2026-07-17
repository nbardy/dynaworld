# Dynamic Foam Sync Boundary

## Context

The trainer-landscape cleanup is still intentionally narrow: move repeated
helper primitives into shared modules only when the behavior contract is
identical, and leave experiment math, schemas, checkpoints, images, and videos
local.

This slice continued the `train_devices` and Dynamic Foam report-helper cleanup.

## Changes

- Routed `verify_powerfoam_4k_trainability.py` through
  `train_devices.sync_torch_device(device)` instead of a local `sync()` helper.
- Routed `verify_powerfoam_raytrace_real_view_alpha.py` through
  `sync_torch_device(rays.device)` instead of a raw MPS synchronize call.
- Fixed a stale import exposed by the raytrace CLI help path:
  `verify_powerfoam_clean_init_coverage.py` now imports
  `report_artifacts.load_report_json(...)` directly instead of importing a
  removed `load_json` helper through `verify_powerfoam_paper_acceptance.py`.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  so the docs reflect the Dynamic Foam sync and clean-init report-reader
  routing.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py \
  research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py \
  research_experiments/dynamic_foam/verify_powerfoam_clean_init_coverage.py \
  src/train/train_devices.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train:research_experiments/dynamic_foam \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_4k_trainability.py --help
```

Passed.

```bash
rtk env PYTHONPATH=src/train:research_experiments/dynamic_foam:third_party/powerfoam-metal \
  .venv/bin/python research_experiments/dynamic_foam/verify_powerfoam_raytrace_real_view_alpha.py --help
```

Initially failed on the stale `load_json` import; passed after the direct
`load_report_json(...)` fix.

```bash
rtk uv run --with pytest python -m pytest \
  tests/test_dynamic_foam_report_artifacts.py \
  tests/test_powerfoam_direct.py::test_atomic_torch_save_preserves_existing_checkpoint_on_failure \
  tests/test_train_devices.py -q
```

Passed: 15 tests.

## Handoff

The current useful cleanup posture remains:

- Keep trainer and experiment loops separate.
- Keep unifying repeated primitives: device/sync, artifact writes, report
  readers, W&B submit/finish, CLI bootstrap, and registry dispatch.
- Do not delete compatibility wrappers unless `rg` proves no live imports and
  a focused runtime smoke covers the call path.
