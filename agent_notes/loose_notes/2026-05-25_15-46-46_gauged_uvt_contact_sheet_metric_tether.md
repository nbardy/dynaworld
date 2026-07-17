# Gauged UVT Contact-Sheet Metric Tether

## Context

The previous real-video media tether proved that the measured live-cache route
and cadence full-rebuild route wrote byte-identical contact sheets. That was
useful, but slightly too shallow: a stale or malformed image artifact could pass
if it happened to match between policies.

This pass tightened the contract in
`research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py`.

## Current Model

The media artifact is now treated as a structured two-row signal, not just a
PNG blob:

- row 0 is target frames,
- row 1 is prediction frames,
- horizontal and vertical gutters are `2px`,
- frame count is inferred from the saved contact-sheet dimensions,
- target/pred pixel MSE is computed from the artifact itself,
- artifact MSE must match `final_full_rgb_loss` within `2.5e-3`.

The tolerance is intentionally larger than exact float noise because the contact
sheet is 8-bit PNG while the trainer payload loss is float-domain. The largest
observed artifact-vs-payload delta after regeneration is
`0.001525666420389149`.

## Evidence

Regenerated:

```text
outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json
outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json
```

Important media summary fields:

```text
all_contact_sheet_layouts_valid = true
all_contact_sheet_metrics_match_payload = true
all_contact_sheet_rows_nontrivial = true
max_contact_sheet_payload_loss_abs_delta = 0.001525666420389149
max_contact_sheet_target_pred_mse_delta = 0.0
min_contact_sheet_target_std = 0.14441643529730494
min_contact_sheet_pred_std = 0.07265247844694266
max_abs_contact_sheet_delta = 0
max_abs_loss_curve_delta = 0.0
```

The top-level audit now surfaces the same facts as:

```text
real_video_multiscene_media_max_contact_sheet_payload_loss_delta
real_video_multiscene_media_max_contact_sheet_target_pred_mse_delta
real_video_multiscene_media_min_contact_sheet_target_std
real_video_multiscene_media_min_contact_sheet_pred_std
```

## Tests

Passed:

```bash
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py
.venv/bin/python -m py_compile research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py -q
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_real_video_multiscene_media_tether_report.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_real_video_multiscene_media_tether/summary.json
.venv/bin/python research_experiments/star_uvt_feature_tubes/projective_goal_progress_audit.py --verify-report outputs/benchmarks/2026-05-25_star_uvt_projective_goal_progress_audit/summary.json --verify-current-inputs
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 uv run --with pytest python -m pytest tests/test_star_uvt_projective_goal_progress_audit.py tests/test_star_uvt_projective_real_video_multiscene_media_tether_report.py -q
```

## Decision Implication

The media tether is now a stronger artifact proof:

```text
live cache == cadence at PNG bytes
and
PNG rows are valid target/pred rows
and
PNG-derived target/pred loss agrees with trainer payload
```

This still does not close broad real-scene quality acceptance. It only removes a
small but real loophole in the saved-media proof.
