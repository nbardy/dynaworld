# Fast-Mac v13a Temporal Recompute State

Date: 2026-05-10

## Summary

Created isolated variant:

```text
third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state
```

Lineage is `variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin`.
The package, custom op namespace, setup metadata, and Metal kernel names were
renamed so v13a imports independently:

- Python package: `torch_gsplat_bridge_v13a_temporal_recompute_state`
- Torch op namespace: `torch.ops.gsplat_metal_v13a_temporal_recompute_state`
- Metal source: `csrc/metal/gsplat_v13a_temporal_recompute_state_kernels.metal`

The implemented recompute path is Python/autograd-level and leaves C++/Metal
math unchanged. `RasterConfig.backward_state_strategy` now accepts:

- `"save"`: copied v11 behavior.
- `"recompute"`: forward saves empty placeholders for tile metadata; backward
  reruns `bin` and the selected `render_*_forward_state` kernel to recreate
  tile metadata before calling the inherited saved-state backward kernel.

## Autograd State Today

The v11/v13a `"save"` path stores these tensors on `ctx`:

- `perm`
- `means_flat`
- `conics_flat`
- `colors_flat`
- `opacities_flat`
- `depths_b`
- `meta_i32`
- `meta_f32`
- `meta_host_i32`
- `meta_host_f32`
- `active_tile_ids`
- `tile_counts`
- `tile_offsets`
- `binned_ids`
- `tile_stop_counts`
- `overflow_tile_ids`
- `overflow_tile_offsets`
- `overflow_sorted_ids`

The persistent tile-state target is:

- `active_tile_ids`: int32 active tile list, empty on direct path.
- `tile_counts`: int32 `[total_tiles]`.
- `tile_offsets`: int32 `[total_tiles + 1]`.
- `binned_ids`: int32 fixedbin storage `[total_tiles * max_fast_pairs]`.
- `tile_stop_counts`: int32 `[total_tiles]`.

For fixedbin, `binned_ids` is the main memory item because it scales with
`total_tiles * max_fast_pairs`, not with actual emitted pairs.

## Recompute Boundary

The exact recomputable state is everything produced from sorted per-splat inputs
and metadata:

```text
tile_counts, tile_offsets, binned_ids = bin(...)
active_tile_ids = _make_active_tile_ids(...) when active mode was selected
tile_stop_counts = render_fast_forward_state(...) or render_active_forward_state(...)
```

Backward still needs the sorted per-splat tensors (`means_flat`, `conics_flat`,
`colors_flat`, `opacities_flat`) and metadata. This v13a path does not yet try to
recompute sorted inputs from original inputs and `depths_b`; that would be a
separate larger change because backward also needs the same sorted arrays for
the backward kernels.

Overflow state is still scaffolded as empty tensors because this fixedbin fork
raises on overflow before backward. A future overflow-capable recompute variant
would need to recreate `_gather_overflow_segments(...)` outputs in backward too.

## Memory/Compute Tradeoff

Expected memory saving for `"recompute"` is the persistent tile-state tensors
above. For the timing probe below (`B=2`, `H=W=128`, `tile_size=16`,
`total_tiles=128`, `max_fast_pairs=2048`), dropping only `binned_ids` removes
`128 * 2048 * 4 = 1,048,576` bytes, plus small per-tile int32 arrays.

The cost is an extra `bin` launch sequence and an extra forward-state render in
backward. That second forward-state render also temporarily allocates image and
alpha outputs in backward to obtain sorted `binned_ids` and `tile_stop_counts`.
So this reduces long-lived autograd saved state, not necessarily instantaneous
backward scratch allocation.

## Commands And Results

Build:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Result: success; built
`torch_gsplat_bridge_v13a_temporal_recompute_state/_C.cpython-311-darwin.so`.

Syntax smoke:

```bash
.venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state/torch_gsplat_bridge_v13a_temporal_recompute_state/rasterize.py \
  third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state/setup.py
```

Result: success.

Import/contract smoke:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state \
  .venv/bin/python <inline save-vs-recompute MPS smoke>
```

Result:

```text
mps_available True
has_ops True
off out_max_abs 0.0
off alpha_max_abs 0.0
off color_grad_max_abs 0.0
on out_max_abs 0.0
on alpha_max_abs 0.0
on color_grad_max_abs 0.0
```

Timing probe:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/v13a_temporal_recompute_state \
  .venv/bin/python <inline B=2 G=4096 F=32 H=128 W=128 timing probe>
```

Result:

```text
active=off strategy=save avg_ms=13.353 best_ms=12.014 worst_ms=14.584 total_pairs=19187 max_pairs=190 active_tiles=128 selected_active=False
active=off strategy=recompute avg_ms=16.026 best_ms=15.161 worst_ms=17.719 total_pairs=19187 max_pairs=190 active_tiles=128 selected_active=False
active=on strategy=save avg_ms=15.240 best_ms=14.462 worst_ms=16.081 total_pairs=19187 max_pairs=190 active_tiles=128 selected_active=True
active=on strategy=recompute avg_ms=19.437 best_ms=18.386 worst_ms=20.611 total_pairs=19187 max_pairs=190 active_tiles=128 selected_active=True
```

Observed overhead on this small probe:

- Direct path: `16.026 / 13.353 = 1.20x`.
- Active path: `19.437 / 15.240 = 1.28x`.

## Shared Renderer Integration

Main-thread follow-up wired v13a into `src/train/renderers/fast_mac.py` as:

```json
"fast_mac": {
  "feature_variant": "v13a_temporal_recompute_state",
  "backward_state_strategy": "recompute"
}
```

The default `backward_state_strategy` is `"save"`, so existing configs remain
unchanged. The new knob is only consumed by v13a.

Shared dispatch smoke:

```bash
GSP_FAST_CAP=4096 GSP_FEATURE_CAP=64 PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train \
  .venv/bin/python <inline shared-dispatch smoke>
```

Result:

```text
v13a_temporal_recompute_state save (1, 32, 32, 32) (1, 32, 32) 0.7124469876289368
v13a_temporal_recompute_state recompute (1, 32, 32, 32) (1, 32, 32) 0.7124469876289368
```

## Remaining Work

- Run a target-shape memory probe with real trainer tensors. This note estimates
  saved-state bytes, but does not include a measured MPS peak-memory trace.
- Decide whether the extra backward forward-state render is acceptable for
  temporal training shapes where fixedbin `binned_ids` dominates saved state.
- A larger follow-up could recompute sorted per-splat tensors too, but that is
  higher risk because it changes the saved tensors needed for unsort and the
  backward kernel inputs.
