# Dynamic Gauge Rendering Boundary

## Goal

Continue trainer modularization by moving Dynamic Gauge Foam render-argument
parsing and full-sequence eval rendering out of the trainer loop file.

## Change

- Added `src/train/dynamic_gauge_rendering.py`.
- Moved `render_kwargs(...)` into
  `dynamic_gauge_rendering.dynamic_gauge_render_kwargs(...)`.
- Moved the no-grad per-frame full-video eval render loop into
  `dynamic_gauge_rendering.render_dynamic_gauge_sequence(...)`.
- Updated `src/train/train_dynamic_gauge_foam.py` to import and call those
  helpers.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  to record the new boundary.

## Why This Boundary

Dynamic Gauge rendering is lane-specific, but it is still pure rendering policy:
config render keys become model kwargs, and eval renders are stitched into
RGB/alpha/depth tensors. Keeping that separate matches the existing
`dynamic_powerfoam_rendering.py` shape while leaving optimizer groups, losses,
metrics payloads, checkpointing, and media policy in the trainer.

## Validation Plan

- Compile the Dynamic Gauge trainer and rendering helper.
- Run focused Dynamic Gauge tests.
- Run a tiny helper smoke for `render_dynamic_gauge_sequence(...)`.
- Search to confirm the trainer no longer defines local `render_kwargs(...)` or
  `render_all(...)`.
- Run whitespace and diff checks on touched files.

## Validation Results

- `rtk .venv/bin/python -m py_compile src/train/train_dynamic_gauge_foam.py src/train/dynamic_gauge_rendering.py tests/test_dynamic_gauge_foam.py` passed.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_dynamic_gauge_foam.py -q` passed: `1 passed`.
- Tiny helper smoke passed for `dynamic_gauge_render_kwargs(...)` and
  `render_dynamic_gauge_sequence(...)` on a two-frame 8px model.
- `rtk rg -n "def render_kwargs|def render_all|dynamic_gauge_render_kwargs|render_dynamic_gauge_sequence" ...` shows no trainer-local `render_kwargs(...)` or `render_all(...)` definitions.
- Touched-file trailing-whitespace scan passed.
- `rtk git diff --check -- src/train/train_dynamic_gauge_foam.py CODE_ORGANIZATION.md TODO/trainer_landscape_unification.md` passed for tracked touched files.
