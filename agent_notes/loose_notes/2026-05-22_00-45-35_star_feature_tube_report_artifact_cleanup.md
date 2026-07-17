# STAR Feature-Tube Report Artifact Cleanup

## Context

The previous helper-import cleanup routed one STAR report/prototype cluster
through `report_artifacts`, but a scan still found direct `.write_text(...)`
calls across the STAR feature-tube profile/matrix scripts. This slice continued
that cleanup without changing shader math or trainer behavior.

## Change

Routed remaining STAR feature-tube report artifacts through shared helpers:

- `targetgrid_render_mode_trainer_matrix.py`
- `sparse_forward_scale_matrix.py`
- `sparse_forward_timing_repeat.py`
- `star_uvt_sparse_forward_profile.py`
- `direct_feature_mode_matrix.py`
- `tile_slot_accumulator_budget.py`
- `sparse_forward_batched_step_benchmark.py`
- `sparse_forward_batched_target_vjp_profile.py`
- `support_birth_split_sweep.py`
- `sparse_visual_loss_vjp_profile.py`
- `star_uvt_logit_handoff_rgb_vjp_profile.py`
- `star_uvt_targetgrid_vjp_bridge_profile.py`
- `compare_compact_visual_vjp_gate.py`
- `run_alpha_background_ablation.py`

The scripts now use `write_report_json(...)`, `write_report_text(...)`, or
`train_artifacts.write_text(...)` for durable JSON/markdown/text artifacts.
Explicit log-file handles and CSV streaming remain local because those are not
the same artifact-write contract.

Two stale helper-boundary bugs surfaced during dry-run validation and were
fixed:

- `targetgrid_render_mode_trainer_matrix.py` no longer references missing
  `_mode_to_kernel(...)`; it uses
  `star_uvt_render_modes.backward_mode_for_feature_render_mode(...)`.
- `star_uvt_targetgrid_vjp_bridge_profile.py` no longer imports missing
  `_mode_to_backward(...)`; it also uses the render-mode helper directly.
- `support_birth_split_sweep.py` now accepts output bases outside the repo root
  by using a display-path helper instead of unconditional `relative_to(ROOT)`.

## Validation

Passed:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/targetgrid_render_mode_trainer_matrix.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_scale_matrix.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_sparse_forward_profile.py \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  research_experiments/star_uvt_feature_tubes/tile_slot_accumulator_budget.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_step_benchmark.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py \
  research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  research_experiments/star_uvt_feature_tubes/compare_compact_visual_vjp_gate.py \
  research_experiments/star_uvt_feature_tubes/run_alpha_background_ablation.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_artifacts.py tests/test_star_uvt_report_artifacts.py -q

git diff --check
```

Focused pytest passed with `9 passed`.

Help checks passed for the patched CLIs, including the two scripts that had
stale helper imports. Dry-run artifact checks wrote JSON/markdown or manifests
under `/tmp` for:

- target-grid render-mode matrix
- sparse-forward timing repeat
- sparse-forward scale matrix
- direct feature-mode matrix manifest
- support birth/split sweep

Targeted scan now returns no direct `.write_text(...)` calls under
`research_experiments/star_uvt_feature_tubes`.

## Remaining

This completes the STAR feature-tube report artifact-write cleanup surface, not
the overall trainer modularization goal. The next stronger evidence should be a
real W&B-enabled trainer smoke or benchmark using the shared train/runtime
interfaces, followed by config curation.
