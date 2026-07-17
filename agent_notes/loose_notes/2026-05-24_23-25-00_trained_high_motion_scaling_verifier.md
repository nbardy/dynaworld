# Trained High-Motion Scaling Verifier

## Context

The Gauged UVT thread had two trained high-motion scaling artifacts:

```text
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json
outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
```

The docs used them as evidence that a trained STAR UVT smoke checkpoint can be
compiled into a reusable sensor-time interval atlas whose non-pixel work grows
slower than per-frame replay. That was good evidence, but it was still a prose
claim. A future continuation could easily quote a stale or malformed JSON.

## Change

Added a report-contract verifier to:

```text
research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py
```

The exported functions are:

```text
verify_trained_high_motion_trace_scaling_report(report) -> list[str]
assert_trained_high_motion_trace_scaling_report(report) -> None
```

The benchmark CLI also accepts:

```text
--verify-report outputs/.../summary.json
```

This validates an existing artifact without rerunning training or Metal timing.

## Contract

The verifier requires:

- `status == "ok"` and benchmark name matches.
- the source high-motion video exists.
- training pass, loss decrease, and zero tile overflow.
- at least two `trained_checkpoint` frame rows.
- trained checkpoint rows are fallback-free.
- trained checkpoint trace count stays constant.
- interval trace entries grow slower than dense per-frame tile-pair work.
- final trained interval/dense tile ratio is in `(0, 1)` and decreases.
- if a per-frame replay baseline exists, the final shared interval route beats
  final per-frame replay in interval entries and trace count.
- if final timing is present for both routes, shared interval forward/backward
  must be finite, positive, and faster than per-frame replay.

This does not prove the final paper-scale claim. It proves the saved smoke
artifacts satisfy the exact sublinear-work evidence they are cited for.

## Verification

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_trained_high_motion_trace_scaling_benchmark.py -q

6 passed in 12.96s
```

Both saved artifacts verified:

```text
PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json

verified outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling/summary.json

PYTHONPATH=src/train .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_trained_high_motion_trace_scaling_benchmark.py \
  --verify-report outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json

verified outputs/benchmarks/2026-05-24_star_uvt_projective_trained_high_motion_trace_scaling_64px_128t/summary.json
```

## Next

The next gate is no longer just "check the JSON by eye." It is:

1. run the verifier for any new trained high-motion scaling artifact,
2. scale beyond the current 64px/128t smoke,
3. preserve the same contract while increasing tube count, resolution, frame
   count, and training duration,
4. only then decide whether the residual cell growth/fallback pattern justifies
   oblique/fiber halfspace cells.
