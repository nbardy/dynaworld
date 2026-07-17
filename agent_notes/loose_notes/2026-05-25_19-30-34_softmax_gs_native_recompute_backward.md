# Softmax-GS Native Recompute Backward

## Context

The active Softmax-GS plan called for replacing the old Python/Torch recompute
backward scaffold before running real trains. The prior state used Metal
forward but recomputed the full differentiable image in Python during
`backward()`, which made `softmax_gs_enabled=true` trainability real but too
slow and too far from the shader architecture.

## What Changed

Implemented a first native backward cut for
`third_party/fast-mac-gsplat/variants/v5_softmax_gs/`:

- Added `render_fast_backward_softmax_recompute(...)` as a separate custom op.
- Added a Metal kernel `tile_fast_backward_softmax_recompute`.
- The kernel recomputes per-pixel Softmax-GS scalar state inside Metal and
  performs the manual VJP for means, conics, colors, opacities, and depths.
- Python training now routes `softmax_gs_enabled=true` through the native
  `_RasterizeProjectedGaussiansV5` autograd path instead of
  `_RasterizeProjectedGaussiansSoftmaxGSTorchBackward`.
- Overflow tiles are deliberately not claimed for enabled native backward yet;
  Python raises if an enabled backward hits overflow. The current tiny smokes
  stay in fast tiles.

This is a native recompute bridge, not the final K-limited tape. Complexity is
too high for large scenes, but it proves the scalar VJP and removes the Python
autograd loop from the standard tiny shader-route smoke.

## Evidence

Build:

```bash
( cd third_party/fast-mac-gsplat/variants/v5_softmax_gs
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Focused Softmax-GS/reference gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_reference.py tests/test_softmax_gs_metal_forward.py -q
```

Result:

```text
11 passed in 11.42s
```

Full focused gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q
```

Result:

```text
20 passed in 12.59s
```

One-step enabled smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_smoke_32_2f_64splats.jsonc
```

Result:

```text
initial loss = 0.4373
step 1 loss = 0.4270
tqdm = 8.59s/it including first-use compile
training complete
```

Five-step enabled diagnostic:

```text
initial loss = 0.4370
step losses = 0.4224, 0.4169, 0.4362, 0.4482, 0.4445
tqdm mean = 2.10it/s
```

Same-session five-step no-op diagnostic:

```text
initial loss = 0.4381
step losses = 0.4496, 0.4487, 0.4372, 0.4467, 0.4324
tqdm mean = 1.62it/s
```

These are local mechanical smokes with W&B disabled/offline. They are not
baseline rows and should not be added to `BASELINES.md`.

## Interpretation

This moves Softmax-GS from "Metal forward plus Python recompute backward" to
"Metal forward plus Metal recompute backward" for fast no-overflow tiles. The
tiny smoke is materially faster than the old one-step recompute scaffold
(`28.35s/it` previously versus `8.59s/it` including compile here), and the
5-step row shows the native path can train for a few steps.

The remaining engineering gap is still real:

- no efficient K-limited forward tape yet;
- no enabled overflow backward yet;
- no learned per-Gaussian beta/gamma/boundary-shape parameters;
- no matched W&B quality row.

## Next Work

1. Decide whether the native recompute bridge is acceptable for one short
   matched quality diagnostic.
2. If not, implement the efficient tape before quality rows.
3. If yes, run a small matched dynamic-GS W&B quality row with vanilla/no-op/
   enabled Softmax-GS under the same split and report it as diagnostic only
   unless it clears the promotion criteria.

## Follow-up: 64px/4f Matched Offline W&B Diagnostic

I ran the next small matched diagnostic after confirming `WANDB_API_KEY` was
unset. W&B ran in offline mode so the run still produced local media/artifacts
without requiring auth.

Configs:

- `src/train_configs/local_mac_softmax_gs_noop_diagnostic_64_4f_128splats_10step.jsonc`
- `src/train_configs/local_mac_softmax_gs_enabled_diagnostic_64_4f_128splats_10step.jsonc`

No-op control:

```text
offline run = wandb/offline-run-20260525_193712-1lra1t7t
initial loss = 0.4330
step losses = 0.4226, 0.4157, 0.4461, 0.4456, 0.4115, 0.4449, 0.4207, 0.4079, 0.4423, 0.4177
tqdm mean = 2.80s/it
media = final images/videos under files/media/
```

Enabled native recompute:

```text
offline run = wandb/offline-run-20260525_193830-fu0df3ks
initial loss = 0.4339
step losses = 0.4339, 0.4455, 0.4373, 0.4414, 0.4322, 0.4457, 0.4449, 0.4308, 0.4578, 0.4413
tqdm mean = 1.27s/it
media = final images/videos under files/media/
```

Interpretation:

This is useful shader-route evidence because enabled Softmax-GS now trains at
the 64px/4f/128-splat shape and logs media. It is not a quality promotion:
source-view loss is noisy over 10 steps, enabled ends worse than no-op on final
loss, there is no heldout claim, and the native backward is still recompute
rather than K-limited tape.

Next:

Do not jump to STAR or WorldFoam from this. The next shader work is either
efficient K-limited tape or enabled overflow backward, then a larger explicitly
budgeted matched quality row.
