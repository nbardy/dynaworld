# Gauge Checkpoint Boundary

## Goal

Continue the trainer modularization cleanup by removing one more live duplicated
checkpoint-persistence path from the Gauge Fields trainers without changing
their experiment-specific payload schemas.

## Change

- Added `research_experiments/gauge_fields/common.write_checkpoint(...)`.
- The helper resolves relative paths with the existing Gauge common
  `resolve_dynaworld_path(...)` and writes through
  `checkpoint_utils.atomic_torch_save(...)`.
- Routed `research_experiments/gauge_fields/train.py` material-surfel
  `checkpoint.pt` through the helper.
- Routed `research_experiments/gauge_fields/train_splat_baseline.py`
  free-dynamic 3DGS `checkpoint.pt` through the helper.
- Removed the local `resolve_dynaworld_path(...)` copy from
  `train_splat_baseline.py`; it now imports the Gauge common resolver.
- Added `tests/test_gauge_common.py` to lock the shared checkpoint helper
  boundary.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/gauge_fields/common.py \
  research_experiments/gauge_fields/train.py \
  research_experiments/gauge_fields/train_splat_baseline.py \
  tests/test_gauge_common.py
```

Passed.

```bash
PYTHONPATH=src/train:research_experiments/gauge_fields uv run --with pytest python -m pytest \
  tests/test_gauge_common.py tests/test_gauge_incidence.py -q
```

Passed: `12 passed in 1.00s`.

## Handoff

This is a narrow persistence-boundary cleanup. It does not claim Gauge training
quality, convergence, or result-schema unification. The next useful cleanup is
another live-file scan for remaining local artifact/path/checkpoint helpers, or
a focused smoke if a touched trainer loop itself changes.
