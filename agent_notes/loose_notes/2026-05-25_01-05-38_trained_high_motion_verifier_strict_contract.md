# Trained High-Motion Verifier Strict Contract

## Context

The Gauged UVT Trace Atlas goal needs evidence that a trained high-motion
checkpoint can compile into reusable sensor-time interval traces rather than
quietly replaying framewise work. The existing benchmark had useful numbers,
but its verifier still accepted too much stale or internally inconsistent
evidence.

Files changed:

- `research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py`
- `tests/test_star_uvt_projective_trained_high_motion_trace_scaling_benchmark.py`
- Gauged UVT docs and clean-thread handoff notes.

## Current Model

The report is credible only if the saved rows jointly prove these contracts:

- training actually changed the checkpoint: pass, loss decrease, zero tile
  overflow;
- frame counts are explicit, positive, increasing, and every trained/per-frame
  row covers the same frame set;
- interval rows are fallback-free and ratio fields match recomputed
  `interval_trace_entries / dense_*` values;
- learned/projected motion is nonzero, so constant topology is not a static
  scene accident;
- opacities stay in `[0, 1]`;
- timing rows include nonzero gradient signals for coeff, opacity, and color;
- saved summaries are recomputed from rows, not trusted as copied text;
- every matched per-frame replay row loses to the shared interval route on
  interval entries, trace count, and timing when timing is present.

## Evidence

Verification commands already run in this continuation:

```text
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  tests/test_star_uvt_projective_trained_high_motion_trace_scaling_benchmark.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trained_high_motion_trace_scaling_benchmark.py -q
```

Result:

```text
8 passed in 27.66s
```

All three saved trained high-motion artifacts verified by CLI:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_96px_256t_cap256/summary.json
```

The dependent shared-work audit suite also passed:

```text
20 passed in 16.90s
```

## Decision Implications

This strengthens the "known camera program -> shared sensor-time trace atlas"
claim for trained high-motion scenes. Future agents should treat this verifier
as a gate before citing trained high-motion scaling, because it checks the
actual row evidence used by the shared-work audit.

The goal is still active, not complete. The next higher-confidence move is a
larger trained artifact with repeated timings and a more direct comparison
against the production trainer path, then deciding whether oblique/fiber
halfspace cells are justified by remaining fallback or cell-growth numbers.

## Open Questions

- Do the timing wins persist under longer training and larger image sizes?
- Does the same strict report contract survive richer WorldFoam/instance cells?
- Are depth-plane slope gradients needed for stability, or can the current
  trace geometry stay geometry-only during this phase?
