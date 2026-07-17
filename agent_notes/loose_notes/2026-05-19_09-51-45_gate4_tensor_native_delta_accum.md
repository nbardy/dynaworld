# Gate4 Tensor-Native Delta Accumulation

## Context

After the track-MSE shader fork came back functionally correct but slower than
the existing sample-thread fused-MSE path, I moved back to the endpoint-record
path. The warm Metal endpoint-record shader is already STAR-shaped; the
remaining bottleneck is host-side endpoint-row / delta-record materialization.

The native Gate4 delta builders already return tensors, but the Python wrapper
converted every native chunk to Python lists with `.tolist()`, extended Python
containers, then rebuilt tensors at the end. That copy path is exactly the
wrong shape for the native compiler lane.

## Change

- Added a tensor-native accumulator in
  `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`.
- Native cut-array, cut-prep, and sorted-delta chunks now append adjusted
  tensor slices directly into chunk lists and concatenate once at the end.
- Packed native outputs keep `base_record_i32` and `change_record_i32` as
  tensors too.
- Python-list fallback remains for the no-native path and for pure-Python
  packed-record materialization.

This is not a shader change by itself; it is the record compiler fork feeding
the fast endpoint-record shader.

## Validation

Passed compile:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py
```

Focused parity passed:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v
```

Full compiler and promotion wrapper tests passed before the final annotation
cleanup:

- Gate4 compiler unit: `8/8`
- framegroup16 promotion wrapper unit: `46/46`

Integration smoke:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 24 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_tensor_native_sorted_emitted_smoke_2f_render16_site24.json
```

Result: `status=ok`, gradients nonzero, parameters updated, outputs finite.
The smoke also confirmed the sorted-native/emitted-packed flags were effective
in the payload.

## Timing status

Do not cite the smoke timing. The artifact reports
`benchmark_environment.status=contended`, with unrelated high-CPU `ai_trader`
pytest/joblib/quote-shadow jobs and a hot `MTLCompilerService`.

The useful status is: tensor-native accumulation preserves endpoint-record
parity and integrated train/eval behavior. A clean promotion ladder is still
needed before claiming setup-speed improvement.
