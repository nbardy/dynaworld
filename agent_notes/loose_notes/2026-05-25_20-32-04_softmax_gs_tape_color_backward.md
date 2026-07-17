# Softmax-GS Tape-Backed Color Backward

Date:
    2026-05-25

Context:
    The bounded top-K contribution tape existed in the `v5_softmax_gs` Metal
    ABI, but native training still used recompute for every backward component.
    The next useful step was to make at least one backward slice consume the
    tape directly without weakening the exact recompute route.

What changed:
    Added `render_softmax_tape_color_backward(...)`, a Metal op that contracts:

    ```text
    dL/dcolor[g] += selected_weight[pixel, slot] * dL/dpixel[pixel]
    ```

    for every selected `(pixel, slot)` in the bounded tape. The TokenGS
    fast-mac config now passes `softmax_gs_tape_k` through to the
    `v5_softmax_gs` `RasterConfig`. When `softmax_gs_tape_k > 0`, the autograd
    forward saves the bounded tape and backward replaces the color-gradient
    slice with this tape contraction. Geometry, opacity, and depth still use
    the native recompute bridge.

Files touched:

```text
src/train/renderers/fast_mac.py
src/train_configs/local_mac_softmax_gs_enabled_tapecolor_diagnostic_32_2f_64splats_5step.jsonc
third_party/fast-mac-gsplat/variants/v5_softmax_gs/torch_gsplat_bridge_v5_softmax_gs/rasterize.py
third_party/fast-mac-gsplat/variants/v5_softmax_gs/csrc/bindings.cpp
third_party/fast-mac-gsplat/variants/v5_softmax_gs/csrc/shared/common.h
third_party/fast-mac-gsplat/variants/v5_softmax_gs/csrc/metal/gsplat_metal.mm
third_party/fast-mac-gsplat/variants/v5_softmax_gs/csrc/metal/gsplat_v5_softmax_gs_kernels.metal
tests/test_softmax_gs_metal_forward.py
```

Evidence:

```text
( cd third_party/fast-mac-gsplat/variants/v5_softmax_gs
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Build passed.

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_metal_forward.py -q -k 'full_tape_color_backward'

2 passed, 6 deselected in 2.90s
```

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q

28 passed in 6.70s
```

Post-shader train:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_tapecolor_diagnostic_32_2f_64splats_5step.jsonc

softmax_gs_tape_k = 8
initial loss       = 0.4382
final loss         = 0.4165
W&B disabled by config
```

Interpretation:
    This is real backward tape consumption, but only for color gradients. It is
    exact when `softmax_gs_tape_k` covers every active contributor and bounded
    otherwise by the residual contract for unit-range features. The remaining
    expensive part is scalar geometry/opacity/depth VJP, which still replays
    prefixes through the native recompute bridge.

Next:
    Extend the bounded tape with selected scalar rows and move
    geometry/opacity/depth VJP off O(K^2) replay. Do not launch larger
    Softmax-GS quality rows, or port the idea to STAR/WorldFoam, until that
    scalar path is in place.
