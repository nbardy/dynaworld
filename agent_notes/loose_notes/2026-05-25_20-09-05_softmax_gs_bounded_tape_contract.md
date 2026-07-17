# Softmax-GS Bounded Tape Contract

Date:
    2026-05-25

Context:
    The Softmax-GS shader route now has native recompute backward for fast and
    overflow tiles, but that bridge is intentionally not the final efficient
    route. The next renderer-lane question is how to avoid O(K^2) per-pixel
    recompute while keeping an auditable approximation/error contract.

What changed:
    Added a reference-only bounded contribution tape:

    ```text
    softmax_gs_bounded_contribution_tape(...)
    ```

    It calls the exact contribution tape, selects the largest final
    contribution weights, returns those selected rows in front-to-back ray
    order, and exposes:

    ```text
    residual_weight = final_alpha - selected_weights.sum()
    ```

Current model:
    The shader should lower toward a small per-pixel K tape containing the
    dominant final color contributors plus a residual mass. The selected rows
    keep enough identity/order information for backward, while the residual
    gives us an explicit approximation budget instead of a silent quality
    cliff.

Bound:
    If omitted feature values are in `[0, 1]`, then dropping the residual
    contributors bounds each output-channel absolute error by
    `residual_weight`.

Evidence:

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_softmax_gs_reference.py -q

11 passed in 3.00s
```

```text
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_fast_mac_depth_signal.py \
  tests/test_softmax_gs_reference.py \
  tests/test_softmax_gs_metal_forward.py \
  tests/test_fast_mac_feature_background.py -q

24 passed in 10.42s
```

Interpretation:
    This does not make Softmax-GS fast yet. It makes the target for the fast
    path precise: lower the bounded tape into `v5_softmax_gs` Metal/ABI and
    replace the native O(K^2) recompute bridge.

Decision implication:
    Do not spend STAR UVT or WorldFoam engineering time on Softmax-GS yet. The
    next useful work is the Metal bounded-tape lowering, followed by a matched
    dynamic-GS heldout/source quality row.
