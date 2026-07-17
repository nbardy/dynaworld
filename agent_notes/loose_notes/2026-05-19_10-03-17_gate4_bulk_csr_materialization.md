# Gate4 Bulk CSR Materialization

## Context

The clean benchmark preflight is still blocked by unrelated `ai_trader` CPU
jobs, so I continued the endpoint-record setup compiler instead of making a
noisy promotion claim.

After the vectorized affine fit and direct event-mask path, the remaining
obvious Python work was still candidate CSR materialization. The fast endpoint
path had vectorized masks but then walked each row/slab in Python to extract
candidates, sort them, extend ids, and append coeff rows.

## Change

- Added `_candidate_csr_from_per_track_event_masks(...)` in
  `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`.
- For `layout="per-track"` + `sample_validation="skip"`, candidate CSR is now
  built by:
  - collecting all `(row, slab, boundary)` hits with `np.nonzero`
  - sorting globally by row/slab plus either boundary id or slab-mid-depth keys
  - using `np.bincount`/`cumsum` for row offsets
  - gathering depth coefficients directly from `all_boundary_coeffs`
- The old Python set/list path remains for tiled layouts and full validation.

This keeps the row ordering contract: row outer, slab inner, candidate sorted
inside each row-slab.

## Validation

Passed:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
```

Focused high-cap parity passed:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v
```

Full suites passed:

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
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_bulkcsr_tensornative_sorted_emitted_smoke_2f_render16_site24.json
```

Result: `status=ok`, gradients nonzero, parameters updated, outputs finite.

Setup split from the smoke's train row:

- `build_endpoint_record_sequences_s`: `0.0375s`
- `build_gate4_affine_endpoint_tape_s`: `0.0245s`
- `build_gate4_endpoint_delta_replace_tape_s`: `0.0129s`

The run is still `benchmark_environment.status=contended`, so do not cite this
as a speed promotion. It is a strong local smoke that the setup compiler path is
now much leaner and still semantically equivalent under the parity tests.

## Next

The next clean gate is still:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --wait-for-benchmark-environment-ok \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```
