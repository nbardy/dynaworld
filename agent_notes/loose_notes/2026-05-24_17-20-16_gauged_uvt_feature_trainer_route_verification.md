# 2026-05-24 17:20 +07 - Gauged UVT feature-trainer route verification

## Context

Heartbeat continuation for Gauged UVT Trace Atlas. The preserved goal remains:
compile 4D spacetime primitives through a known camera program into reusable
sensor-time traces for fast rasterization across time, with clean derivatives
and shared compute/memory/backward work across frames.

## Current State

The tree is ahead of the previous handoff. The compatible UVT tube producer is
now routed through the real STAR UVT feature trainer for the exact RGB-width
case:

- `feature_uvt.projective_interval.enabled=true` is accepted for
  `feature_dim=3`.
- The trainer locks UV spatial precision to the backend `sigma_px`, so the
  compatible-tube producer can lower the UVT quadratic to interval cell traces.
- Motion, temporal precision, opacity, and feature gradients remain live.
- Feature color renders through `ProjectiveCellIntervalTrainerState`.
- Total alpha renders through a second white-trace interval atlas so the
  existing alpha-background/colorizer objective stays valid.
- The route is still full-frame/autograd-only and rebuilds the atlas each step.

The lower-level projective atlas now also carries temporal opacity envelopes
via `opacity_time_coeffs`, consumed by the CPU/Torch reference path and native
interval Metal forward/backward. Metal VJP returns
`grad_opacity_time_coeffs`.

## Verification

Targeted producer/config/trainer-route slice:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_uvt_producer.py \
  tests/test_star_uvt_render_configs.py -q
```

Result:

```text
16 passed in 14.02s
```

Focused projective plus interval-gated trainer suite:

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
102 passed in 27.37s
```

`--collect-only` confirms this exact focused set currently collects 102 tests.
The docs had a stale 107-pass count; corrected to the verified 102-pass count.

## Next

The next useful gate is not another render flag. It is cache ownership:

1. Define when the feature trainer rebuilds the projective atlas versus when
   `ProjectiveCellIntervalTrainerState.refresh()` is enough.
2. Attach budget/fallback telemetry to the trainer row for this route.
3. Keep the exact-route constraints loud until anisotropic footprints and
   pixel-varying depth have native representation.
