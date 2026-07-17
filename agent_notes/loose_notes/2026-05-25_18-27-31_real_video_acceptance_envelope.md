# Real-Video Acceptance Envelope

## Context

The active STAR UVT / gauged UVT goal is still broader than any single saved
artifact: fast 2D rasters across time from 4D spacetime primitives, with clean
forward/backward behavior and shared projection/support/binning/visibility work
over time. Before this pass, the goal-progress audit verified many individual
real-video rows but did not have one artifact that consolidated the functional,
frame-scaling, quality, and media evidence while preserving the "not complete"
scope.

## Current Model

The useful claim is now an acceptance envelope, not broad acceptance:

```text
current proof = focused real-video envelope + explicit remaining gaps
not = full real-scene renderer/trainer completion
```

The new verifier lives at:

```text
research_experiments/star_uvt_feature_tubes/projective_real_video_acceptance_envelope_report.py
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
```

It reads eight already-saved reports:

```text
trainer_matrix
extended_trainer_matrix
frame_scaling_matrix
extended_frame_scaling_diagnostic
quality_tether
extended_quality_tether
media_tether
extended_media_tether
```

and rejects stale summaries, missing theory scope, source-count loss, underlying
verifier errors, media deltas, quality deltas, support churn, rebuild-ratio
regressions, and any attempt to turn the envelope into completion proof.

## Evidence

Saved envelope summary:

```text
underlying_report_count = 8
functional_scene_count = 5
media_scene_count = 5
all_underlying_verifiers_pass = true
all_functional_rows_pass = true
all_quality_tethers_match = true
all_media_tethers_match = true
max_support_rebins = 0
max_stale_refreshes = 0
max_rebuild_ratio = 0.5
min_quality_psnr_gain = 0.02227306365966797
extended_frame_scaling_expected_timing_failure_count = 2
max_extended_timing_growth_overage = 0.0009153415685994037
strict_timing_win_claimed = false
does_not_prove_completion = true
```

The goal-progress audit now imports this envelope, adds
`real_video_acceptance_envelope` as a proved requirement row, increments the
proved count from 31 to 32, and keeps `full_goal_completion` open.

## Tests And Commands

Focused report tests:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_real_video_acceptance_envelope_report.py \
  tests/test_star_uvt_projective_goal_progress_audit.py -q
```

Result:

```text
49 passed in 9.06s
```

Saved artifact verification:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_real_video_acceptance_envelope_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_acceptance_envelope/summary.json
```

Goal audit current-input verification:

```text
PYTHONPATH=src/train uv run python \
  research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json \
  --verify-current-inputs
```

Both passed.

## Backtracks

This strengthens evidence organization but does not change the theoretical
status of the project. The five-source extended frame-scaling diagnostic still
has expected timing failures, and the envelope explicitly preserves those
failures rather than hiding them. The open goal remains broader real-scene
quality acceptance and a fuller compiled-adjoint trainer replacement.

## Next Falsification Tests

1. Run a broader scene-set acceptance envelope with more source diversity and
   fresh-process timing medians.
2. Add a quality threshold tied to a stronger baseline, not only measured-vs-
   cadence equality.
3. Move from report aggregation to a full trainer substitution test where the
   compiled atlas path is the default path under realistic training cadence.
