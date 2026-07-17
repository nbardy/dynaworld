# Clip Sampling Helper

Date: 2026-05-21 16:41

## Goal Context

The active goal is trainer modularization. The previous slice moved
`prepare_clip` to the data module; this slice pulls the repeated sampling
pattern itself behind a small helper.

## Slice Implemented

Added `src/train/clip_sampling.py`:

- `sample_clip_batch(sequence, train_frame_count=..., frame_sampling=..., device=...)`

The helper centralizes:

1. `temporal_sampling.select_frame_indices(...)`
2. `sequence_data.make_clip(...) -> ClipBatch`

Added `ClipBatch.as_time_batch(...)` to `runtime_types.py` so callers can adapt
typed clips back to the legacy `[1, K]` time tensor without re-shaping by hand.

Updated these trainer paths to call `sample_clip_batch(...)`:

- `Trainer.sample_clip`
- `KnownCameraTrainer.sample_clip`
- `MulticamPrecomputedFeatureImplicitTrainer.sample_multicam_clip`
- the multicam camera-swap sampling branch

The public trainer return shapes are unchanged for now. This is deliberate:
the helper creates a typed internal boundary without forcing all downstream
trainer code to migrate in the same step.

## Validation

Passed:

```bash
PYTHONPATH=src/train:. rtk .venv/bin/python -m py_compile \
  src/train/clip_sampling.py \
  src/train/runtime_types.py \
  src/train/sequence_data.py \
  src/train/train_video_token_implicit_dynamic.py \
  src/train/train_multicam_precomputed_feature_implicit_dynamic.py \
  tests/test_temporal_sampling.py

PYTHONPATH=src/train:. rtk uv run --with pytest python -m pytest \
  tests/test_temporal_sampling.py \
  tests/test_pipeline_helpers.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_pipeline_diagnostics.py \
  tests/test_train_logging.py \
  tests/test_config_factory_helpers.py -q
```

Result: `45 passed in 1.03s`.

Also passed `git diff --check` and trailing-whitespace scan.

Runtime smoke after the final trainer edits:

```bash
PYTHONPATH=src/train WANDB_MODE=offline rtk .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  /tmp/dynaworld_tokengs_1step_smoke.jsonc
```

The temp config was copied from
`src/train_configs/local_mac_overfit_video_token_smoke.jsonc` with
`train.steps` patched from 10 to 1. Result: passed on MPS; step 0 init
diagnostic and the single optimizer step completed. Offline W&B run:
`wandb/offline-run-20260521_164720-pzj6jlzo`.

## Next Refactor Sequence

1. Move trainer internals from legacy `(sequence, clip_frames, clip_times)`
   tuples toward `ClipBatch` where the local blast radius is low.
2. Build the mixed same-view plus heldout-view scheduler on top of typed
   batches, with separate `same_view_recon` and `heldout_view_recon` names.
3. Remove compatibility re-exports only after all local imports have moved.
