# 2026-05-24 16:24 +07 - Gauged UVT compatible UVT producer/state bridge

## Context

Heartbeat continuation for Gauged UVT Trace Atlas. The production config/state
bridge already existed, and the latest tree also had the lower-level
`uvt_tubes_to_projective_trace_cell_atlas(...)` producer for compatible affine
UVT tubes. The missing production-side convenience was a single `src/train`
helper that takes live UVT tensors and returns the refresh/fallback-aware
trainer state.

## Change

Extended `src/train/star_uvt_projective_interval_backend.py`:

- `make_projective_cell_interval_atlas_from_uvt_tubes(...)` now forwards
  optional `primitive_ids` into the producer.
- Added `make_projective_cell_interval_trainer_state_from_uvt_tubes(...)`.
  It compiles compatible UVT tube tensors into `ProjectiveTraceCellTraceAtlas`
  and immediately wraps them in `ProjectiveCellIntervalTrainerState`.

The helper preserves the narrow exactness contract: by default the producer
still rejects anisotropic spatial precision, pixel-varying depth, and residual
temporal opacity envelopes. `temporal_mode="gate"` remains the explicit coarse
interval-gating escape hatch.

## Tests

Targeted producer/config gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py -q
```

Result:

```text
12 passed in 11.34s
```

Focused Gauged UVT gate with the producer suite included:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
98 passed in 27.10s
```

`--collect-only` confirms this focused set currently collects 98 tests.

## Next

Route this producer/state helper through the real STAR UVT feature training
loop. The current trainer still keeps the projective interval backend behind
the explicit producer/routing tripwire, so the next useful change is the actual
render path selection and cache/refresh ownership around the helper, not
another config flag.
