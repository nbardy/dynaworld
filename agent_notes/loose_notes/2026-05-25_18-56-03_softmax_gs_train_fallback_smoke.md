# Softmax-GS Train Fallback Smoke

## Context

The prior Softmax-GS chunk proved the `v5_softmax_gs` Metal forward shader and
no-op train route. The active blocker was trainability with
`softmax_gs_enabled=true`.

## Current Model

Softmax-GS needs a native/tape backward before it can be a real fast renderer
lane. A guarded Torch autograd fallback is still useful because it proves the
enabled compositing law can participate in optimizer steps and lets us run tiny
dynamic-GS smokes while designing the native backward.

## What Changed

- Added `_rasterize_softmax_gs_torch_train(...)` in
  `third_party/fast-mac-gsplat/variants/v5_softmax_gs/torch_gsplat_bridge_v5_softmax_gs/rasterize.py`.
- Training with `softmax_gs_enabled=true` now uses the Torch fallback.
- Eval/no-grad with `softmax_gs_enabled=true` still uses the Metal shader.
- Vanilla/no-op training still uses the existing native backward path.
- Added the matched no-op/active tiny configs:
  - `src/train_configs/local_mac_softmax_gs_enabled_smoke_32_2f_64splats.jsonc`
  - `src/train_configs/local_mac_softmax_gs_noop_smoke_32_2f_64splats.jsonc`
- Extended `tests/test_softmax_gs_metal_forward.py` so the MPS gate checks:
  - same-depth two-splat order invariance;
  - Torch fallback forward agreement with Metal forward;
  - finite gradients for means, scales, quats, opacities, and colors.

## Numerical Backtrack

The first vectorized Torch fallback did not match Metal forward on a random
six-splat case. The problem was not projection or sorting. It was the quadratic
rescale:

```text
scale = (pair_sum - sqrt(discriminant)) / (2 * pair_product)
```

For tiny current absorbance, float32 cancellation can make the numerator zero
on vector MPS. The stable equivalent is:

```text
scale = 2 * target_absorbance / (pair_sum + sqrt(discriminant))
```

The reference, Torch fallback, and Metal forward now use this rationalized form.
For Torch autograd, the discriminant is clamped to `eps` so masked branches do
not leak NaN gradients through `sqrt(0)`.

## Evidence

Focused gate:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q
```

Result:

```text
16 passed
```

Random six-splat forward/gradient check:

```text
Metal-vs-fallback forward_max_abs = 2.980232238769531e-07
finite gradients: means/scales/quats/opacities/colors all true
```

Enabled tiny train smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_smoke_32_2f_64splats.jsonc
```

Result:

```text
Step 0 initialization diagnostic: Loss: 0.4389 recon: 0.4389
Loss: 0.4525 recon: 0.4525
DynamicVideoTokenGSImplicitCamera training complete (W&B disabled).
```

Matched no-op tiny train smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_noop_smoke_32_2f_64splats.jsonc
```

Result:

```text
Step 0 initialization diagnostic: Loss: 0.4381 recon: 0.4381
Loss: 0.4503 recon: 0.4503
DynamicVideoTokenGSImplicitCamera training complete (W&B disabled).
```

Matched 5-step diagnostics:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_noop_diagnostic_32_2f_64splats_5step.jsonc

PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_diagnostic_32_2f_64splats_5step.jsonc
```

Results:

```text
no-op:
    initial loss 0.4376
    step losses 0.4479, 0.4258, 0.4743, 0.4307, 0.4891
    tqdm mean 3.62s/it

enabled:
    initial loss 0.4375
    step losses 0.4331, 0.4381, 0.4483, 0.4335, 0.4885
    tqdm mean 4.85s/it
```

## Interpretation

This proves trainability, not quality. The enabled smoke is about `17.74s/step`
at `32px/2f/64 splats`, while the matched no-op is about `1.01s/step`. That gap
is expected because the enabled path is a dense Torch loop over splats and
pixels.

The 5-step diagnostic narrows the timing gap after warmup/validation overhead,
but it is still too small and noisy for a quality claim. Final losses are nearly
tied (`0.4891` no-op, `0.4885` enabled).

Do not update `BASELINES.md`. These are mechanical smokes with W&B disabled,
not benchmark rows.

## Next Decisions

1. Native/tape backward is the next serious renderer task.
2. A very short slow-fallback ablation is possible, but it should be labeled
   diagnostic-only and not promoted.
3. STAR UVT should still wait. This result only says dynamic-GS Softmax-GS can
   train mechanically; it does not show that Softmax-GS fixes STAR support.
