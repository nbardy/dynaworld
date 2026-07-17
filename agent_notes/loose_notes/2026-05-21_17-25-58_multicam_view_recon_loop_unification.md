# Multicam View Recon Loop Unification

## Context

The active trainer modularization goal is to keep the train code simple while
removing behavior forks that make future experiments drift. After the mixed
same-view plus heldout trainer bridge landed, the multicam trainer still had
separate train-view and heldout-view reconstruction loops with the same
background sampling, alpha/background guard, render call, reconstruction loss,
and preview capture mechanics.

## Change

- Added `MulticamPrecomputedFeatureImplicitTrainer._recon_loss_for_views(...)`.
- `multicam_recon_loss(...)` now delegates to that helper with
  `target_frames=self.multicam_bundle.train_frames` and
  `render_fn=self.render_view_clip`.
- `heldout_recon_loss(...)` now delegates to that helper with
  `target_frames=self.multicam_bundle.heldout_frames` and
  `render_fn=self.render_heldout_view_clip`.
- Distinct loss names and target banks stay outside the helper. The helper only
  owns per-view render/loss mechanics: background sampling, profiling scopes,
  alpha/background guarding, reconstruction-loss accumulation, and optional
  preview capture.

This is deliberately small. It avoids a base-trainer abstraction and keeps the
same-view versus heldout semantics explicit at the public method boundary.

## Validation

Focused plumbing tests:

```bash
PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_multicam_video_data.py \
  tests/test_mixed_same_heldout_trainer.py \
  tests/test_mixed_data_scheduler.py \
  tests/test_rgb_recon_objective.py \
  tests/test_temporal_sampling.py \
  tests/test_pipeline_helpers.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py -q
```

Result after the final docs patch: `69 passed in 1.21s`.

Runtime smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train.py \
  src/train_configs/local_mac_mixed_same_heldout_rgb_pyramid_32_2f_64splats_10step_smoke.jsonc
```

Result: passed on MPS. The first post-extraction run wrote
`wandb/offline-run-20260521_172310-9iwq2eer`; the fresh current-state rerun
wrote `wandb/offline-run-20260521_172710-2xs5airh`.

Visible trace from the first post-extraction run:

- init: `0.5898`
- step 1 same-view: `0.5268`
- step 2 heldout: `0.6086`
- step 3 same-view: `0.5020`
- step 4 heldout: `0.6009`
- step 5 same-view: `0.4871`
- step 6 heldout: `0.6069`
- step 7 same-view: `0.4915`
- step 8 heldout: `0.6054`
- step 9 same-view: `0.5002`
- step 10 heldout: `0.6025`

## Interpretation

This validates trainer wiring and makes the shared reconstruction path less
likely to drift. It does not prove convergence, visual quality, or the loss
math. The next real evidence should be a W&B-enabled mixed run with media and
separate same-view/heldout metrics, or another narrow code slice that removes a
live behavior fork without changing the experiment contract.
