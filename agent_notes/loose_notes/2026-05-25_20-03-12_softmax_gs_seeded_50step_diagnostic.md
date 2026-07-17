# Softmax-GS Seeded 50-Step Diagnostic

Date:
    2026-05-25 20:03:12

Context:
    After adding native Softmax-GS overflow backward, the remaining question was
    whether to spend more STAR/WorldFoam engineering attention on the
    compositor. The previous 10-step matched draws were not cleanly matched
    because TokenGS did not normalize or apply a train seed.

Changes:

- Added `train.seed` defaulting to `17` in `src/train/token_gs_trainer.py`.
- `Trainer.__init__` now calls `torch.manual_seed(int(self.train_cfg["seed"]))`
  before model construction and sampling.
- Added config tests for default and explicit train seeds in
  `tests/test_config_factory_helpers.py`.
- Pinned all Softmax-GS configs to `seed=17`.
- Added seeded 50-step matched configs:
  - `src/train_configs/local_mac_softmax_gs_noop_diagnostic_seed17_64_4f_128splats_50step.jsonc`
  - `src/train_configs/local_mac_softmax_gs_enabled_diagnostic_seed17_64_4f_128splats_50step.jsonc`

Verification:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_config_factory_helpers.py::test_resolve_config_defaults_train_seed \
  tests/test_config_factory_helpers.py::test_resolve_config_accepts_explicit_train_seed -q
```

Result: `2 passed in 3.50s`.

Full helper file follow-up:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_config_factory_helpers.py -q
```

Result: `17 passed in 3.53s`.

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q
```

Result: `21 passed in 7.67s`.

Seeded 50-step no-op control:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_noop_diagnostic_seed17_64_4f_128splats_50step.jsonc
```

Result:

```text
initial loss 0.4338
final loss 0.1467
tqdm mean 1.65it/s
offline run wandb/offline-run-20260525_200015-s04n74di
```

Seeded 50-step enabled Softmax-GS:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_diagnostic_seed17_64_4f_128splats_50step.jsonc
```

Result:

```text
initial loss 0.4338
final loss 0.1512
tqdm mean 1.32it/s
offline run wandb/offline-run-20260525_200101-xd4sm546
```

Interpretation:

- The seeded row is cleaner than the earlier 10-step diagnostics.
- Enabled Softmax-GS is neutral to slightly worse on this tiny source-view
  setup: final train loss `0.1512` vs no-op `0.1467`.
- Do not port Softmax-GS to STAR/WorldFoam from this evidence.
- Next useful shader work remains the efficient K-limited tape. The current
  native recompute bridge is correct enough to test but still not the final
  renderer lane.
