# 2026-05-24 15:22 +07 - Gauged UVT production interval bridge

## Context

Heartbeat continuation for the Gauged UVT Trace Atlas goal: compile 4D
spacetime primitives through a known camera program into reusable sensor-time
traces, with clean derivatives and shared compute/memory/backward work across
time. The previous state had a strong research-harness bridge for projective
interval cells, lifecycle refresh, visibility strata, fallback metadata, and
continuous exposure/rolling reference oracles. The next gate was to begin
moving that bridge into real `src/train` production surfaces.

## Change

Added `src/train/star_uvt_projective_interval_backend.py`.

This module gives the production trainer side a single place to normalize and
validate the projective interval-cell policy:

- `feature_uvt.projective_interval` config defaults.
- `ProjectiveCellIntervalBackendConfig` for refresh, budget, and fallback
  policy.
- `make_projective_cell_interval_trainer_state(...)`, which constructs the
  existing `ProjectiveCellIntervalTrainerState` from a compiled atlas, sensor
  times, and trainer config.

Also wired `star_uvt_feature_config.resolve_config(...)` to normalize the
optional `feature_uvt.projective_interval` section once during config
resolution. This keeps the policy out of ad hoc call-site defaults.

## Tests

Targeted production config and trainer bridge:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
16 passed in 6.11s
```

Expanded focused Gauged UVT gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
79 passed in 13.32s
```

Config-regression pass, because `star_uvt_feature_config` now imports the new
normalizer:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_render_configs.py \
  tests/test_star_uvt_feature_target_adapter.py -q
```

Result:

```text
39 passed in 7.68s
```

## Next

This is not yet a full production STAR UVT projective renderer. It is the
configuration/state bridge. The next implementation step is to add or promote a
real producer that compiles the model/camera program into
`ProjectiveTraceCellTraceAtlas`, then route a production trainer mode through
`make_projective_cell_interval_trainer_state(...)` instead of constructing
harness state directly.
