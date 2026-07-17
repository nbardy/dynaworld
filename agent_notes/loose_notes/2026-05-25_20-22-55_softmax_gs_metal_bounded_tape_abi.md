# Softmax-GS Metal Bounded Tape ABI

Date:
    2026-05-25

Context:
    The short-term Softmax-GS plan called for lowering the bounded top-K
    contribution tape into `v5_softmax_gs` before spending more runs on quality.
    Before this change, the bounded tape existed only in the Torch reference;
    native training still used an O(K^2) recompute bridge for backward.

What changed:
    Added a Metal/Python ABI for the bounded tape:

    ```text
    rasterize_softmax_gs_bounded_tape(...)
        -> selected_ids
        -> selected_weights
        -> residual_weight
        -> final_alpha
    ```

    The fast-tile kernel and overflow kernel both compute the exact final
    top-K contribution weights online. Previous selected weights are scaled by
    each Softmax-GS prefix rewrite, new contributors are inserted by final
    contribution weight, then selected slots are sorted back into ray/global-ID
    order. Residual mass is:

    ```text
    final_alpha - selected_weights.sum()
    ```

Files touched:

```text
third_party/fast-mac-gsplat/variants/v5_softmax_gs/torch_gsplat_bridge_v5_softmax_gs/rasterize.py
third_party/fast-mac-gsplat/variants/v5_softmax_gs/torch_gsplat_bridge_v5_softmax_gs/__init__.py
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
  tests/test_softmax_gs_metal_forward.py -q -k 'bounded_tape'

2 passed, 4 deselected in 6.90s
```

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_metal_forward.py -q

6 passed in 3.51s
```

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q

26 passed in 4.85s
```

Post-shader trainer smokes:

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_diagnostic_32_2f_64splats_5step.jsonc

initial loss 0.4382
final loss   0.4165
W&B disabled by config
```

```text
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_overflow_smoke_32_2f_64splats_2step.jsonc

initial loss 0.4382
step losses  0.4394, 0.4529
W&B disabled by config
```

Interpretation:
    This completes the tape ABI/lowering step, not the final efficient
    Softmax-GS backward. The training path still uses native recompute backward.
    The next real shader task is to make backward consume the bounded tape and
    retire the O(K^2) replay scaffold.

Decision implication:
    Still do not port Softmax-GS into STAR UVT or WorldFoam. The dynamic-GS
    renderer lane needs bounded-tape-backed backward and one honest matched
    heldout/source quality row first.
