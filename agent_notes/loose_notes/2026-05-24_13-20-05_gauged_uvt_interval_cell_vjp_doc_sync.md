# Gauged UVT interval cell VJP doc sync

Date: 2026-05-24

## Context

Heartbeat continuation after the two-lane Gauged UVT fork. The previous handoff
said the next gate was projective atlas-cell trainability or interval-compressed
backward coverage.

## What changed in this pass

I checked the newest working tree and found that the next gate had already
landed in the active files:

```text
render_projective_trace_cell_interval_atlas_metal(...)
direct_backward_projective_trace_cell_interval_atlas_metal(...)
test_projective_cell_trace_interval_atlas_one_step_coeff_training_smoke_if_available
```

The interval-compressed cell path now has:

- spatial tile bins with per-entry `[active_start, active_stop)` intervals
- interval forward Metal render
- interval direct VJP Metal backward
- Torch-autograd parity for color, opacity, and homogeneous coefficients
- one-step coefficient trainability smoke with color fixed

## Verification

Ran the focused projective plus interval-gated trainer suite:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trace.py \
  tests/test_star_uvt_projective_orbit_windows.py \
  tests/test_star_uvt_projective_visibility.py \
  tests/test_star_uvt_projective_binning.py \
  tests/test_star_uvt_projective_correctness.py \
  tests/test_star_uvt_trainer_interval_gated.py -q
```

Result:

```text
48 passed in 4.32s
```

## Docs updated

Updated:

```text
research_notes/gauged_uvt_trace_atlas/GOAL_META_KEY_MATH.md
research_notes/gauged_uvt_trace_atlas/README.md
research_notes/gauged_uvt_trace_atlas/clean_thread_handoff/README.md
```

The next best gate is no longer "add interval backward." It is now:

```text
trainer producer for nontrivial projective/gauge-domain active intervals
or
bridge interval-compressed projective cells into trainer production
```
