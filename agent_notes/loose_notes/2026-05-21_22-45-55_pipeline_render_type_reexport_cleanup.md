# Pipeline Render Type Re-Export Cleanup

## Goal

Tighten ownership after moving render payload dataclasses into
`runtime_types.py`.

## Evidence

`rg` found no real code imports of `RasterizedClip` or `RenderedClip` through
`pipeline.render`. Active code and tests import those payload types from
`runtime_types`.

## Changed

- Removed `RasterizedClip` and `RenderedClip` from `pipeline.render.__all__`.
- Left the internal imports in `pipeline.render` because the module still uses
  the dataclasses in type hints and return construction.

## Validation

Run after this edit:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/pipeline/render.py tests/test_pipeline_helpers.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_rendering_contracts.py tests/test_pipeline_helpers.py -q
```

Passed: `8 passed in 7.08s`.
