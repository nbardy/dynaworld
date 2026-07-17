# RasterizedClip-Only Render Boundary

## Goal

Finish the render-boundary cleanup started by introducing
`render_gaussian_frames_rasterized(...) -> RasterizedClip`. The remaining old
API was `render_gaussian_frames_alpha_aware(...)`, a tuple wrapper returning
`(features, alpha)`.

## Evidence Before Editing

`rg` over the repo, excluding generated outputs/W&B/third-party folders, found
no real code imports of `render_gaussian_frames_alpha_aware(...)`. The only
active import was the compatibility test in `tests/test_rendering_contracts.py`.
Current trainer rendering enters through
`pipeline.render.render_clip_sequence(...) -> RasterizedClip`.

## Changed

- Removed `render_gaussian_frames_alpha_aware(...)` from `src/train/rendering.py`.
- Removed it from `rendering.__all__`.
- Updated the single-frame fast-mac comment to point alpha-capable callers at
  the typed `RasterizedClip` batch wrapper.
- Removed the compatibility test that unpacked the tuple wrapper.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md` to
  record that `render_gaussian_frames_rasterized(...) -> RasterizedClip` is now
  the public alpha-aware batch-render API.

## Validation

Run after this edit:

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/rendering.py tests/test_rendering_contracts.py
```

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_rendering_contracts.py tests/test_pipeline_helpers.py -q
```

Passed: `8 passed in 6.51s`.

## Remaining

`render_gaussian_frames(...)` still exists for tensor-only/legacy callers. That
is fine for now: the active alpha-aware trainer boundary is typed, and deleting
the tensor-only API would require a broader audit of direct renderer users.
