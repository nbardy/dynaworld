# Gauged UVT Bq4 Policy Order

Context:

The Bq4 sequence-order report showed that `reverse_16_to_4` can produce
substep timing bumps even when the high-level `16f` no-first ratio stays below
1.0. That did not say whether the bump follows:

- the measured policy,
- the second execution slot,
- or a more general warm-state/order interaction.

New artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order/summary.json
```

What it runs:

- Warmup block with traced Bq4 `4f` cadence, `4f` measured, `16f` cadence,
  `16f` measured.
- Then two repeats of `16f` `cadence_then_measured`.
- Then two repeats of `16f` `measured_then_cadence`.
- All target runs trace the saved Bq4 `16f` spike step.

Observed facts:

- All expected global steps are traced.
- All chunks include projective interval substep timing.
- Cache/support remains clean.
- Target `paired_ratio_count = 4`.
- Across target pairs:
  - `no_first_bump_count = 1`
  - `projective_total_bump_count = 3`
  - `feature_state_update_bump_count = 3`
  - max no-first ratio `1.7836530508238704`
  - max projective-total ratio `1.7184222253396344`
  - max feature-state-update ratio `1.9605903379413647`

By policy order:

```text
cadence_then_measured:
  measured slot = 1
  no_first_bump_count = 0
  projective_total_bump_count = 1
  feature_state_update_bump_count = 1
  max no_first_ratio = 0.9092050121020188
  max projective_total_ratio = 1.5935737599142148
  max feature_state_update_ratio = 1.6440789864667122

measured_then_cadence:
  measured slot = 0
  no_first_bump_count = 1
  projective_total_bump_count = 2
  feature_state_update_bump_count = 2
  max no_first_ratio = 1.7836530508238704
  max projective_total_ratio = 1.7184222253396344
  max feature_state_update_ratio = 1.9605903379413647
```

Current model:

The warmed timing issue does not reduce to "the second slot is slower." In this
artifact, measured-first is the worse order. The signal is better described as
a policy/order/warm-state interaction. It is still not a reason to change the
fiber/gauge formulation because cache/support are clean and the effect is in
runtime phase timings, not trace correctness.

Backtrack:

Earlier hypothesis:

```text
substep bump is warm-state/launch-order variance that may not affect no-first
```

Status:

partially supported, partially sharpened. Warm state matters, but one warmed
policy-order target does reproduce a no-first bump. So timing acceptance needs
explicit policy/order controls; one small matrix can hide this depending on
case ordering.

Decision implication:

Next timing work should isolate runtime effects, not atlas math:

1. add warmup-discard runs where warmup cases are not included in summary,
2. randomize or alternate policy order in matrix reports,
3. report policy order and execution slot in timing artifacts,
4. consider per-case fresh-process timing if we need acceptance-grade numbers.

Validation:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_policy_order_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_policy_order/summary.json
```

passed.

Focused expanded verifier pack including the policy-order report:

```text
135 passed in 1.09s
```
