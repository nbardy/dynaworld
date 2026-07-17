# Manifest Sequence Sampler Unification

Date: 2026-05-21

## Goal

Continue the trainer modularization goal by removing duplicated same-view
manifest sampling mechanics from trainer classes while keeping the data-contract
distinction between same-view and heldout-view explicit.

## What changed

- Added `sequence_data.ManifestSequenceSampler`.
- The sampler owns manifest entries, eager/lazy loading, cycle/random sampling,
  lazy index cursor state, optional one-worker prefetch, minimum frame-count
  validation, and export-by-index loading.
- `train_video_token_implicit_dynamic.Trainer` now uses the sampler for
  manifest-backed train sequences instead of owning lazy manifest cursors and
  prefetch futures itself.
- `train_mixed_same_heldout_implicit_dynamic.MixedSameHeldoutPrecomputedFeatureTrainer`
  now uses the same sampler for the same-view side, while multicam/heldout
  bundles remain owned by `multicam_video_data.py`.

This is a code-organization move only. It does not claim better convergence or
training quality.

## Validation

- Compile check:
  `PYTHONPATH=src/train rtk .venv/bin/python -m py_compile src/train/sequence_data.py src/train/train_video_token_implicit_dynamic.py src/train/train_mixed_same_heldout_implicit_dynamic.py tests/test_temporal_sampling.py tests/test_mixed_same_heldout_trainer.py`
- Focused pytest:
  `PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest tests/test_temporal_sampling.py tests/test_mixed_same_heldout_trainer.py tests/test_mixed_data_scheduler.py tests/test_multicam_video_data.py tests/test_pipeline_helpers.py tests/test_sequence_data_single_frame.py tests/test_pipeline_diagnostics.py tests/test_train_logging.py tests/test_config_factory_helpers.py -q`
  passed with `63 passed in 1.90s`.
- Checked-in mixed smoke:
  `PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python src/train/train.py src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc`
  passed on MPS at `wandb/offline-run-20260521_171453-7wqptf1i`.

## Remaining

- Keep pushing trainer-specific manifest logic into data/sampler helpers only
  when it preserves same-view versus heldout semantics.
- The mixed trainer still needs a longer W&B-enabled media run before any
  baseline row.
- The F=32/colorizer/random-background inheritance audit remains open before
  using the mixed trainer as the main feature-splatting training harness.
