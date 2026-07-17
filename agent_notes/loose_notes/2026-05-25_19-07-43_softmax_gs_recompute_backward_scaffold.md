# Softmax-GS Recompute Backward Scaffold

## Context

The previous state trained `softmax_gs_enabled=true` through a dense Torch
forward/backward fallback. That proved trainability, but the training forward
was no longer the Metal shader. The next serious lane is native/tape backward,
so I changed the enabled training path to the closer architecture:

```text
forward: Metal `v5_softmax_gs`
backward: recompute differentiable Torch Softmax-GS reference and call autograd
```

This is still not the final native backward. It is a recompute scaffold that
isolates the part that must become a native/tape kernel.

## What Changed

File:
    `third_party/fast-mac-gsplat/variants/v5_softmax_gs/torch_gsplat_bridge_v5_softmax_gs/rasterize.py`

Added:
    `_RasterizeProjectedGaussiansSoftmaxGSTorchBackward`

Behavior:

- custom autograd `forward()` calls the Metal eval path;
- `backward()` detaches/requires-grad on projected inputs;
- backward recomputes `_rasterize_softmax_gs_torch_train(...)`;
- `torch.autograd.grad(...)` returns gradients for means2d, conics, colors,
  opacities, and depths.

## Evidence

Focused Softmax-GS tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_softmax_gs_reference.py -q
```

Result:

```text
7 passed
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
16 passed
```

Enabled one-step smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_smoke_32_2f_64splats.jsonc
```

Result:

```text
initial loss = 0.4384
step 1 loss = 0.4453
tqdm = 28.35s/it
training complete
```

## Interpretation

This is progress toward the native/tape architecture, not a speed improvement.
The one-step smoke is slower than the earlier dense Torch fallback smoke because
it pays both the Metal forward launch and the Torch recompute backward. The
value is architectural: the train forward is now the same Metal forward used
for eval/no-grad, and the remaining slow piece is explicitly the backward
replacement target.

## Next Gate

The native/tape backward should implement the recompute logic without PyTorch:

1. Forward pass per pixel records or recomputes enough prefix state:
   `T`, `accum`, `past_depth`, `past_power`, effective alpha, and correction
   scale.
2. Reverse pass propagates gradients through the Softmax-GS update into
   `power`, `alpha`, color, and opacity.
3. Geometry gradients reuse the vanilla local derivatives from `power`.

Until that exists, keep all enabled Softmax-GS quality rows diagnostic-only.
