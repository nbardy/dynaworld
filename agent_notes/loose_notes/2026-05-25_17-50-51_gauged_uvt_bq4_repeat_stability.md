# Gauged UVT Bq4 Repeat Stability

Context:

The previous Bq4 traced spike rerun showed that the saved no-first-step timing
spike did not reproduce, but one traced `16f` measured run had a higher
projective interval total than cadence. The bump was concentrated in
`feature_state_update_ms`, so the working question became: is this a real
live-update hotspot or another small-sample timing artifact?

New artifact:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability/summary.json
```

What it does:

- Targets only the Bq4 `16f` trace spec from the saved render-forward shape
  report.
- Repeats the cadence/measured traced pair three times with the same fixed seed.
- Captures projective interval substep timing at the expected traced global
  step.
- Keeps the guarded support setup from the prior Bq4 trace rerun.

Observed facts:

- `paired_repeat_count = 3`.
- All expected global steps are traced.
- All chunks include projective interval substep timing.
- Cache/support remains clean: support rebins, stale refreshes, fallback marks,
  and tile overflow are zero.
- `no_first_spike_reproduced_count = 0`.
- `projective_total_bump_count = 0`.
- `feature_state_update_bump_count = 0`.
- Max measured/cadence no-first ratio is `0.45165397508134686`.
- Max measured/cadence projective interval total ratio is
  `0.9101288137358652`.
- Max measured/cadence feature-state-update ratio is `0.7882220153002857`.

Current model:

The one-shot `16f` feature-state-update bump from the prior Bq4 trace rerun is
not persistent under a 16f-only repeat schedule. This weakens the hypothesis
that live-update feature-state-update is an intrinsic hot substep. The stronger
remaining hypothesis is schedule/warm-state variance: the earlier bump may
depend on the mixed `4f` then `16f` sequence, allocator/cache state, command
buffer timing, or MPS launch synchronization state.

Backtrack:

Earlier note:

```text
feature_state_update/live-update phase cost is the next timing target
```

Status:

weakened, not invalidated. The substep can still spike, but the repeat artifact
does not support it as a stable 16f live-update cost.

Decision implication:

Do not change the fiber/gauge/chart math for this timing miss. If we keep
investigating timing, the next focused report should compare:

1. `16f`-only repeats.
2. mixed `4f -> 16f` repeats.
3. reversed `16f -> 4f` repeats.
4. optional warmup-discard repeats.

The falsifier would be persistent measured/cadence substep ratios over `1.0`
under a controlled repeat schedule. The current artifact says we do not have
that yet.

Validation:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_bq4_trace_repeat_stability_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_bq4_trace_repeat_stability/summary.json
```

passed.

Focused expanded verifier pack including the repeat-stability report:

```text
125 passed in 4.74s
```
