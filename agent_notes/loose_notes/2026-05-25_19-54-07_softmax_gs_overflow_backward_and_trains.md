# Softmax-GS Overflow Backward And Fresh Trains

Date:
    2026-05-25 19:54:07

Context:
    Continued the Softmax-GS renderer plan after the native fast-tile recompute
    backward was already working. The next concrete shader gap was enabled
    overflow backward: Python still raised when `softmax_gs_enabled=true`
    touched overflow tiles.

Changes:

- Added a device-ID replay helper and `tile_overflow_backward_softmax_recompute`
  to `third_party/fast-mac-gsplat/variants/v5_softmax_gs/csrc/metal/gsplat_v5_softmax_gs_kernels.metal`.
- Exposed the new op as
  `gsplat_metal_v5_softmax_gs.render_overflow_backward_softmax_recompute`.
- Routed `_RasterizeProjectedGaussiansV5.backward` to add overflow gradients
  for means/conics/colors/opacities/depths when Softmax-GS is enabled.
- Added a forced-overflow unit test:
  `tests/test_softmax_gs_metal_forward.py::test_softmax_gs_overflow_backward_matches_torch_recompute_reference`.
- Added
  `src/train_configs/local_mac_softmax_gs_enabled_overflow_smoke_32_2f_64splats_2step.jsonc`
  with `max_fast_pairs=1` to hit the overflow route through the trainer.

Verification:

```text
( cd third_party/fast-mac-gsplat/variants/v5_softmax_gs
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Build succeeded.

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_metal_forward.py::test_softmax_gs_overflow_backward_matches_torch_recompute_reference -q
```

Result: `1 passed in 7.81s`.

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q
```

Result: `21 passed in 6.91s`.

Forced-overflow trainer smoke:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_overflow_smoke_32_2f_64splats_2step.jsonc
```

Result:

```text
initial loss 0.4374
step losses 0.4775, 0.4486
training complete, W&B disabled
```

Fresh matched 64px/4f/128-splat diagnostics after the overflow shader:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_noop_diagnostic_64_4f_128splats_10step.jsonc
```

No-op result:

```text
initial loss 0.4337
final loss 0.4456
tqdm mean 2.43s/it
offline run wandb/offline-run-20260525_195019-27rj83gw
```

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_diagnostic_64_4f_128splats_10step.jsonc
```

Enabled result:

```text
initial loss 0.4342
final loss 0.4198
tqdm mean 1.60s/it
offline run wandb/offline-run-20260525_195115-tn9t3nby
```

Interpretation:

- Enabled Softmax-GS now has native recompute backward coverage for both fast
  and overflow tiles.
- The old enabled-overflow hard stop is gone.
- The fresh 10-step diagnostic is directionally positive for Softmax-GS, unlike
  the previous tiny draw, but it is still source-view-only and too short to
  promote.
- Next shader work should be the efficient K-limited tape. Larger quality rows
  should not claim representation evidence until the O(K^2) recompute bridge is
  replaced or the run is explicitly budgeted as a slow diagnostic.
