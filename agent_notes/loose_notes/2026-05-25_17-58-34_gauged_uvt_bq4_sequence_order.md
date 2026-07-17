# Gauged UVT Bq4 Sequence Order

Context:

The Bq4 repeat-stability artifact weakened the hypothesis that
`feature_state_update_ms` is a persistent `16f` live-update hotspot. It only
tested a `16f`-only schedule, though. The previous one-shot bump happened in a
mixed report that ran multiple frame sizes. This note records the follow-up:
does mixed frame-size ordering affect traced `16f` substep timing?

New artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order/summary.json
```

What it runs:

- Two repeats of `mixed_4_to_16`.
- Two repeats of `reverse_16_to_4`.
- For each sequence/frame, runs cadence and measured policies.
- Traces the saved Bq4 spike step for `4f` and `16f`.
- Keeps the same guarded support settings as the prior Bq4 trace diagnostics.

Observed facts:

- All expected steps are traced.
- All chunks include projective interval substep timings.
- Cache/support remains clean.
- `paired_16f_ratio_count = 4`.
- Across all `16f` pairs, `no_first_bump_count = 0`.
- Max `16f` no-first ratio is `0.45600195672964483`.
- Across all `16f` pairs, `projective_total_bump_count = 2`.
- Across all `16f` pairs, `feature_state_update_bump_count = 3`.

By sequence:

```text
mixed_4_to_16:
  max 16f projective_total_ratio = 0.9606946419165872
  max 16f feature_state_update_ratio = 1.0006466493572015
  max 16f no_first_ratio = 0.45600195672964483

reverse_16_to_4:
  max 16f projective_total_ratio = 1.844612661591509
  max 16f feature_state_update_ratio = 1.73336471126077
  max 16f no_first_ratio = 0.3432623079420896
```

Current model:

The high-level measured/cadence no-first timing win survives the sequence-order
test. The substep-level projective interval timings are nevertheless sensitive
to ordering/warm-state effects, especially when `16f` is first in the reversed
sequence. That points away from fiber/chart math and toward MPS/Metal launch,
allocator, command-buffer, or warm-cache phase variance.

Backtrack:

Earlier hypothesis:

```text
feature_state_update/live-update may be a persistent hot substep
```

Status:

weakened. It is not persistent in `16f`-only repeats and is order-sensitive in
the sequence test. The artifact supports treating substep timing ratios as
phase diagnostics, not acceptance gates by themselves.

Decision implication:

Do not redesign the trace atlas or gauges because of this timing branch. If
timing work continues, the next useful report should isolate launch/warm-state
effects directly:

1. add a warmup discard before traced measurement,
2. alternate policy order (`measured` before `cadence`),
3. record per-case wall-clock order and maybe MPS sync boundaries,
4. compare one-process sequence runs against fresh-process single-case runs.

Validation:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_sequence_order_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_sequence_order/summary.json
```

passed.

Focused expanded verifier pack including the sequence-order report:

```text
130 passed in 2.89s
```
