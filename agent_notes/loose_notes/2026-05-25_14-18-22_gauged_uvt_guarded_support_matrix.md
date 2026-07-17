# Gauged UVT Guarded Support Matrix

Goal context: keep advancing the Gauged UVT Trace Atlas evidence without
marking the full objective complete. The remaining gap after active-set
distribution was broader real-video/trainer acceptance, especially whether the
support guard mechanism had durable checked-in evidence rather than loose
notes.

What changed:

- Regenerated real-video guarded trainer artifacts for the checked-in
  high-motion clip with `support_guard_policy=slack_budgeted`,
  `support_stale_tail_alpha_epsilon=0.001`, and guard paddings
  `0.25/0.5/1.0/2.0`.
- Added
  `research_experiments/star_uvt_feature_tubes/projective_real_video_guarded_support_matrix_report.py`.
- Added
  `tests/test_star_uvt_projective_real_video_guarded_support_matrix_report.py`.
- Wired the aggregate into
  `research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py`
  as `real_video_guarded_support_matrix`.
- Updated the top audit artifact:
  `outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json`.

Key numbers:

- Default unguarded measured support rebins: `9`.
- Guarded measured support rebins: `0`.
- Guarded measured stale refreshes: `0`.
- Guarded artifact count: `4`.
- Total matrix artifact count: `5`.
- Measured rows: `15`.
- Max guarded measured/cadence no-first-step ratio: `0.5895631229975254`.
- Max guarded measured/cadence rebuild ratio: `0.5`.
- Max guarded loss delta: `0.0`.
- Max guarded tile count: `18`.

Interpretation:

This is not broad real-scene quality acceptance, and it is not a full trainer
replacement proof. It does promote the support-lifecycle piece from a loose
claim to a checked-in matrix: on the high-motion real-video trainer route, a
small slack-budgeted support guard removes measured support rebins/stale
refreshes while preserving the live-cache reuse contract.

Verification:

```text
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest \
  tests/test_star_uvt_projective_goal_progress_audit.py \
  tests/test_star_uvt_projective_real_video_guarded_support_matrix_report.py -q

39 passed in 0.99s
```

```text
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_guarded_support_matrix_report.py \
  --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json

verified outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_guarded_support_matrix/summary.json
```

```text
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py \
  --verify-current-inputs

verified /Users/nicholasbardy/git/gsplats_browser/dynaworld/outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json against current inputs
```

Current top audit state:

- proved requirements: `23`
- open requirements: `1`
- `is_goal_complete=false`

The open gap is intentionally still broad real-scene/trainer acceptance beyond
focused synthetic/high-motion probes.
