# Helper Import And STAR Report Cleanup

## Context

Continuing the trainer-code modularization goal, I took a live-file cleanup
slice instead of expanding abstractions:

- remove remaining generic helper lookups through the large Token-GS trainer
- route another coherent STAR UVT report/prototype family through the shared
  report/artifact helpers
- keep kernel/prototype-specific math local

## Change

- `src/benchmarks/trainer_phase_benchmark.py` now imports
  `trainer_class_for_config(...)` from `trainer_registry` and delegates device
  synchronization to `train_devices.sync_torch_device(...)`.
- `src/train_scripts/train_single_video_pretrain_300_64f.sh` now imports
  `decoded_token_count_from_model_config(...)` from `render_dispatch`, not from
  `train_video_token_implicit_dynamic.py`.
- `research_experiments/star_uvt_feature_tubes/report_artifacts.py` now inserts
  both the Dynaworld root and `src/train` on `sys.path`, so directly launched
  report/prototype scripts can import both `research_experiments.*` and shared
  train helpers through one bootstrap point.
- These STAR UVT report/prototype scripts now use shared report JSON/text
  writers instead of local parent-directory creation plus direct text writes:
  `star_uvt_vjepa_bridge_audit.py`,
  `dense_alpha_failure_diagnostic.py`,
  `dense_feature_tube_prototype.py`,
  `star_uvt_vjepa_vs_gaussian_comparison.py`,
  `alpha_only_visibility_profile.py`,
  `visibility_support_bridge_prototype.py`,
  `visibility_support_birth_split_prototype.py`,
  `star_uvt_feature1_wholegraph_profile.py`, and
  `target_cache_budget.py`.
- The dense feature-tube prototype also now uses
  `train_devices.resolve_torch_device(...)` and `sync_torch_device(...)`.

## Validation

Passed:

```bash
bash -n src/train_scripts/train_single_video_pretrain_300_64f.sh

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python -m py_compile \
  src/benchmarks/trainer_phase_benchmark.py \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py \
  research_experiments/star_uvt_feature_tubes/alpha_only_visibility_profile.py \
  research_experiments/star_uvt_feature_tubes/visibility_support_bridge_prototype.py \
  research_experiments/star_uvt_feature_tubes/visibility_support_birth_split_prototype.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/target_cache_budget.py

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py \
  --gate0-benchmark --frames 2 --height 8 --width 8 --tubes 3 --feature-dim 4 \
  --steps 2 --chunk-size 1 --device cpu \
  --out-json /tmp/dense_feature_tube_prototype_shared_helpers_2step.json

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_devices.py tests/test_star_uvt_report_artifacts.py -q

git diff --check
```

Help surfaces also executed cleanly for the patched report/prototype scripts and
`src/benchmarks/trainer_phase_benchmark.py`.

The tiny dense feature-tube CPU gate wrote
`/tmp/dense_feature_tube_prototype_shared_helpers_2step.json` with
`pass=true`. Focused pytest passed with `10 passed`.

## Remaining

This is still cleanup progress, not goal completion. Remaining useful work:

- route the remaining STAR profile/matrix scripts that still write JSON or
  markdown directly
- run a W&B-enabled mixed/trainer smoke that proves the shared paths log media
  and losses in a real train loop
- curate STAR UVT configs into keepers versus historical probes
