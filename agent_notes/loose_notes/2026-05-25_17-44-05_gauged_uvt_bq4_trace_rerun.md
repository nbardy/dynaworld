# Gauged UVT Bq4 Trace Rerun

Goal context: continue the Gauged UVT Trace Atlas line where 4D spacetime
primitives compile through a known camera program into reusable sensor-time
traces. The active implementation question was whether the five-source
real-video frame-scaling timing miss meant the math needed richer charts/fibers,
or whether it was a render-forward timing artifact inside the existing
projective interval path.

What changed:

- Added `trace_global_steps` plumbed through the multiscene trainer matrix
  harness so selected global steps can capture chunk traces.
- Added optional projective interval substep timing to the projective interval
  render autograd path. When chunk tracing is enabled it records:
  `feature_state_update_ms`, `feature_render_ms`,
  `alpha_state_update_ms`, `alpha_render_ms`, and
  `projective_interval_render_ms`.
- Added the Bq4 traced spike-step rerun report:
  `research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_rerun_report.py`.
- Added verifier coverage:
  `tests/test_star_uvt_projective_real_video_multiscene_bq4_trace_rerun_report.py`
  and a matrix-harness test ensuring `trace_global_steps` stays in config.

Artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun/summary.json
```

Result:

- The rerun targets the Bq4 `4f` and `16f` saved render-forward spike steps
  selected from the render-forward shape artifact.
- All expected global steps are traced.
- All traced chunks include projective interval substep timing.
- Cache/support remains clean: support rebins, stale refreshes, fallback marks,
  and tile overflow stay zero.
- The saved Bq4 spike does not reproduce at no-first-step level:
  `traced_bq4_spike_reproduced = false`.
- Measured/cadence no-first ratios are `0.4538476088322886` and
  `0.5785517503959672`.
- Projective interval substep totals are mixed: measured/cadence ratios are
  `0.5054386427773483` at `4f` and `1.2736600499593582` at `16f`.
- Feature-state-update measured/cadence ratios are `0.44341185194975186` and
  `1.250134158419622`, making that phase the most interesting remaining
  substep.

Interpretation:

The original saved Bq4 failure is best treated as small-step render-forward
timing variance rather than a failure of the fiber/gauge/chart math. However,
the traced 16f live-update projective interval still has a real-looking
feature-state-update cost bump. The next useful experiment is repeat/stability
profiling around feature-state-update/live-update phases, possibly with more
substep counters for state-update buffer work, synchronization, and Metal
launch boundaries.

Validation:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_rerun_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_rerun/summary.json
```

passed.

Focused expanded verifier pack:

```text
120 passed in 5.33s
```
