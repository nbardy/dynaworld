# Gate4 Direct Mask CSR

## Context

After vectorizing the affine ray fit and splitting setup timers, the smoke
showed `build_gate4_affine_endpoint_tape_s` was still the dominant endpoint
setup cost. The remaining obvious Python cost was candidate CSR materialization:
the real endpoint path (`layout="per-track"`, `sample_validation="skip"`) still
allocated per-row Python sets and updated them once per track/slab.

## Change

- Added `_compiled_slab_event_mask_from_coeffs(...)` in
  `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`.
- For the real endpoint path (`per-track` + `skip` validation), Gate4 now
  builds candidate CSR rows directly from vectorized `[track, boundary]` slab
  event masks.
- Preserved the older Python-set path for tiled layouts and full validation,
  because full validation compares candidate sets against per-frame reference
  events.
- Preserved the existing candidate order and row order:
  row outer, slab inner, and slab-mid-depth sorting for per-track rows.

## Validation

Passed:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
```

Passed focused high-cap parity:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v
```

Passed broader suites:

- Gate4 compiler unit: `8/8`
- framegroup16 promotion wrapper unit: `46/46`

Integrated smoke:

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
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_vectorfit_maskcsr_tensornative_sorted_emitted_smoke_2f_render16_site24.json
```

Result: `status=ok`, gradients nonzero, parameters updated, outputs finite.
The artifact remains timing-contended, so it is not a speed claim. It is still
useful as a local before/after smoke on setup instrumentation:

- train `build_endpoint_record_sequences_s`: `0.1217s`
- train `build_gate4_affine_endpoint_tape_s`: `0.1057s`
- train `build_gate4_endpoint_delta_replace_tape_s`: `0.0160s`

The previous vector-fit-only smoke had train `build_endpoint_record_sequences_s`
around `0.1757s`, also contended. Treat the drop as encouraging, not as a
promotion result.

## Next

The clean promotion command remains:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --wait-for-benchmark-environment-ok \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

Do not cite warm-step speed or setup speed for this fork until that promotion
gate runs with `benchmark_environment.status=background/ok`.
