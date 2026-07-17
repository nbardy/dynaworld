# Softmax-GS Selected Scalar Tape VJP

Date:
    2026-05-25 20:51:31 Asia/Ho_Chi_Minh

Context:
    The Softmax-GS plan's next blocker was geometry/opacity/depth VJP still
    using the O(K^2) native recompute bridge. Color gradients already consumed
    the bounded top-K tape when `softmax_gs_tape_k > 0`.

Implementation:
    Added a selected-row scalar tape backward path to
    `third_party/fast-mac-gsplat/variants/v5_softmax_gs/`.

    New Metal pieces:
    - `replay_softmax_to_selected_slot(...)`
    - `softmax_tape_scalar_backward(...)`

    New C++/Torch op:
    - `gsplat_metal_v5_softmax_gs.render_softmax_tape_scalar_backward(...)`

    Autograd routing:
    - `torch_gsplat_bridge_v5_softmax_gs/rasterize.py` now uses
      `render_softmax_tape_scalar_backward` plus
      `render_softmax_tape_color_backward` whenever
      `softmax_gs_enabled=true` and `softmax_gs_tape_k > 0`.
    - `softmax_gs_tape_k=0` still uses the native recompute bridge.

Contract:
    The selected scalar backward replays only the selected tape IDs per pixel.
    If the tape covers every active contributor, the path is exact against the
    Torch reference. If K is bounded, it is an explicit selected-contributor
    approximation. This is a real move off the hidden full-prefix replay, but
    K/residual quality now matters.

Build:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v5_softmax_gs
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Tests:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_metal_forward.py -q -k 'full_tape_backward'
```

Result:
    `2 passed, 6 deselected in 5.87s`

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q
```

Result:
    `28 passed in 4.17s`

Train diagnostics:

1. 5-step K=8 tape-scalar smoke:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_tapecolor_diagnostic_32_2f_64splats_5step.jsonc
```

Result:
    initial `0.4382`, final `0.4190`, W&B disabled by config.

2. 50-step K=8 tape-scalar source-view diagnostic:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_tapescalar_diagnostic_seed17_64_4f_128splats_50step.jsonc
```

Result:
    initial `0.4338`, final `0.2026`, tqdm mean `1.38it/s`, offline run
    `wandb/offline-run-20260525_204628-sk2fc3ne`.

Interpretation:
    K=8 selected scalar tape is too lossy for this 64px/4f/128-splat row. It
    is much worse than the earlier seeded no-op `0.1467` and enabled recompute
    `0.1512` rows.

3. 50-step K=16 tape-scalar source-view diagnostic:

```bash
PYTHONPATH=src/train WANDB_MODE=offline GSP_TAPE_CAP=16 .venv/bin/python \
  src/train/train_video_token_implicit_dynamic.py \
  src/train_configs/local_mac_softmax_gs_enabled_tapescalar_k16_diagnostic_seed17_64_4f_128splats_50step.jsonc
```

Result:
    initial `0.4338`, final `0.1472`, tqdm mean `3.19it/s`, offline run
    `wandb/offline-run-20260525_204816-oip27eka`.

Interpretation:
    K=16 recovers the old tiny source-view bracket while using the selected
    scalar tape path. It is close to no-op `0.1467` and a little better than
    enabled recompute `0.1512`, but this remains source-view-only and is not a
    quality promotion.

Decision:
    Do not port Softmax-GS to STAR or WorldFoam from this evidence. The next
    dynamic-GS step is residual/tape-coverage diagnostics or a small K sweep
    (`8/16/32`) before a matched heldout/source quality row.
