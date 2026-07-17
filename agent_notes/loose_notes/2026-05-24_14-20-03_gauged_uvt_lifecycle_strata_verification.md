# 2026-05-24 14:20 +07 - Gauged UVT lifecycle and strata verification

## Context

Heartbeat continuation for Gauged UVT Trace Atlas. The preserved goal remains:
compile 4D spacetime primitives through a known camera program into reusable
sensor-time traces for fast rasterization across time, with clean derivatives
and maximal compute/memory/backward reuse so non-pixel costs grow sublinearly
with frame count.

## What I found

The working tree already contains the next trainer-lifecycle increment beyond
the previous two-lane fork:

- `render_projective_cell_interval_atlas_metal_backward(...)` gives the trainer
  harness an autograd-facing interval-compressed projective cell path.
- `ProjectiveCellIntervalTrainerState` owns atlas/config/times/refresh cadence,
  renders through the interval path, and refreshes after optimizer steps.
- Support drift and depth-order drift are detected and repaired without
  replacing optimizer-owned coefficient tensors.
- Ambiguous near-tie visibility can be marked as `visibility_ambiguous_depth`,
  rejected by the Metal fast path, measured via fallback stats, and rendered by
  the CPU/Torch reference fallback with live-depth sorting.
- Before fallback, sampled visibility-stratum splitting now converts crossing
  depth order into stable time-run cells.

## Verification

Ran the focused projective plus interval-gated trainer gate from the dynaworld
root:

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
60 passed in 9.85s
```

## Doc sync

`research_notes/gauged_uvt_trace_atlas/clean_thread_handoff/README.md` now
points a clean thread at the actual next gate: promote the harness bridge/state
into a production STAR UVT backend/producer, then set refresh thresholds, cache
ownership, and the fast/fallback scheduler.

## Next

1. Move the projective interval-cell producer out of focused harness tests and
   into the real STAR UVT training backend.
2. Define production staleness thresholds: support coverage, order crossings,
   stratum count, fallback fraction, and recompile cadence.
3. Decide the mixed production fallback path: refit/split gauge domains first,
   then tile-local live sort or a depth-bin/k-buffer scheduler for the cells
   that remain ambiguous.
