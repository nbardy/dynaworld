# Owner-Run Factorized Packed-Delta Shader Fork

Goal: port the positive coeff-factorization probe into the actual WorldFoam
Gate4/owner-run packed-delta Metal path, removing resident `delta_coeff_f16`
from the new hot tape mode.

## Implemented

- Added native op:
  `endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only`.
- Added Metal helper:
  `wf2_endpoint_record_factorized_cut_depth(boundary_f32, track_ray_coeff_f32, ...)`.
- Added Metal kernel:
  `wf2_endpoint_record_delta_replace_factorized_packed_framegroup16_recompute_mse_vjp_direct_atomic_rgb_only_tensor`.
- Added Python wrapper/export in
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/`.
- Added train/eval mode:
  `owner-run-delta-packed-factorized-recompute-fused-mse-nomid`.

The new train/eval mode keeps packed owner-run base/change topology, builds
per-track linear ray coefficients, stores `boundary_f32 + track_ray_coeff_f32`,
and intentionally does not store `delta_coeff_f16`.

## Validation

Build:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
```

Focused tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
```

Result: 5 tests passed.

The moving-ray parity test now checks both:

- `owner-run-delta-packed-recompute-fused-mse-nomid`
- `owner-run-delta-packed-factorized-recompute-fused-mse-nomid`

against `owner-run-fused-mse-nomid`.

Follow-up regression added in the same test file:

- prepares factorized tapes at `2/4/8f`
- asserts `delta_coeff_f16` is absent
- asserts `boundary_f32` and `track_ray_coeff_f32` are resident
- asserts schema coeff storage and resident coeff storage stay constant from
  `2f` to `8f`
- asserts selected storage grows sublinearly versus frame count

Updated focused suite:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
```

Result: 6 tests passed.

## Functional Ladder

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_owner_run_delta_packed_factorized_recompute_nomid_ladder_2_4_8_16_render16_site8_contended.json
```

Command used `2/4/8/16f`, render16, site8, `steps=2`, `warmup_steps=1`,
manual VJP, moving rays, slow-owner-run source.

Result: all rows `status=ok`.

Storage/timing snapshot:

| frames | total ms | backward ms | selected bytes | resident bytes | coeff storage | coeff resident | topology bytes | train PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 3.334 | 2.892 | 42,444 | 42,500 | 25,136 | 24,576 | 17,308 | 13.37 |
| 4 | 3.248 | 2.837 | 55,812 | 55,876 | 25,136 | 24,576 | 30,676 | 13.41 |
| 8 | 10.605 | 8.770 | 70,160 | 70,240 | 25,136 | 24,576 | 45,024 | 13.34 |
| 16 | 4.627 | 3.736 | 79,576 | 79,688 | 25,136 | 24,576 | 54,440 | 13.53 |

Scales over `8x` frames:

- selected schema storage: `1.875x`
- selected resident storage: `1.875x`
- resident coeff storage: `1.0x`
- total step: `1.388x`
- backward: `1.292x`

## Caveat

The ladder was environment-contended by unrelated `ai_trader` Python processes
and `MTLCompilerService`, so the speed numbers are functional evidence only.
Do not use this as the clean promotion timing. Re-run the same ladder with
`--require-benchmark-environment-ok` before claiming speed parity or a STAR UVT
comparison.

A later environment check was still blocked by an unrelated
`scripts/report_btc15m_rl_feature_policy_sweep.py` Python process at about
`98%` CPU, so clean timing remained unavailable.

## Current Decision

The factorized shader fork is correctness-green and storage-positive. This is
the first actual Metal path in the lane where coefficient residency is constant
over frame count. The remaining blocker is clean promotion timing at the
matched `2/4/8/16f` and higher-cap shapes.

## Follow-up: high-cap storage regression

Clean timing is still blocked by unrelated high-CPU `ai_trader`/pytest Python
processes, so no promotion ladder was run.

I added a 24-site high-cap regression to
`research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py`.
The test prepares the actual factorized train/eval tape at `2f` and `8f`,
asserts `delta_coeff_f16` is absent, asserts `boundary_f32 + track_ray_coeff_f32`
are resident, checks factorized resident coeff storage stays constant across
frames, and compares against the dense packed path at the same 24-site shape.

That exposed and fixed a metric-accounting bug: for the factorized mode,
`endpoint_record_coeff_mps_resident_storage_bytes` used to count only keys with
`coeff` in the name, so it counted `track_ray_coeff_f32` but not `boundary_f32`.
The accounting now treats the factorized resident coeff representation as
`boundary_f32 + track_ray_coeff_f32`.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
```

Results:

- py_compile passed
- delta-packed test file: 3 tests passed in 184.740s
- combined focused suite: 7 tests passed in 208.102s

## Follow-up: schema storage breakdown

Clean timing remained blocked by unrelated pytest/Kalshi/Toto CPU work. I added
schema-side storage attribution for train/eval rows instead of running another
noisy ladder. `train_selected_tape_schema_storage_by_key` now reports base
offsets, track-change offsets, change-frame rows, change offsets, base packed
records, change packed records, extra storage, and either `factorized_coeff_f32`
or `delta_coeff_f16`.

The high-cap regression now asserts the factorized schema storage is fully
attributed, contains `factorized_coeff_f32`, does not contain `delta_coeff_f16`,
and still shows `change_record_packed` growing across frames. That last check is
intentional: coefficient storage is fixed; the next storage fork should target
change-record/index growth.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --frame-counts 2 --render-size 16 --site-count 4 --near 0.0 --far 3.5 --density 8.0 --invalid-epsilon 1.0e-7 --transmittance-threshold 1.0e-4 --steps 1 --warmup-steps 0 --optimizer-mode manual-vjp --tape-mode owner-run-delta-packed-factorized-recompute-fused-mse-nomid --endpoint-record-source slow-owner-run --out-json /tmp/worldfoam_factorized_schema_smoke.json
```

Results:

- py_compile passed
- combined focused suite: 7 tests passed in 230.123s
- schema smoke wrote `/tmp/worldfoam_factorized_schema_smoke.json` with
  `train_selected_tape_schema_storage_by_key` present and `status=ok`; timing is
  not promotable because the artifact records `benchmark_environment.status=contended`

## Follow-up: int16 metadata projection

I added projected int16 accounting for the remaining metadata arrays that are
range-eligible: base offsets, change offsets, track-change offsets, and
change-frame rows. This does not change the Metal layout yet; it brackets the
next storage fork and emits explicit row fields:

- `train_selected_tape_schema_i16_meta_projection_eligible`
- `train_selected_tape_schema_i16_meta_projected_storage_bytes`
- `train_selected_tape_schema_i16_meta_projected_storage_savings_bytes`
- `train_selected_tape_schema_i16_meta_projected_storage_by_key`
- `train_selected_tape_schema_i16_meta_projection_fields`

The high-cap regression now asserts the projection is eligible, that
`change_frame_i32` becomes `change_frame_i16`, that the projected breakdown sums
to the projected total, and that projected schema storage is below current
schema storage.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py research_experiments/world_foam_lane2/train_eval_owner_run_tape.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_highcap_storage_removes_dense_coeff16 -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --frame-counts 2 --render-size 16 --site-count 4 --near 0.0 --far 3.5 --density 8.0 --invalid-epsilon 1.0e-7 --transmittance-threshold 1.0e-4 --steps 1 --warmup-steps 0 --optimizer-mode manual-vjp --tape-mode owner-run-delta-packed-factorized-recompute-fused-mse-nomid --endpoint-record-source slow-owner-run --out-json /tmp/worldfoam_factorized_i16_projection_smoke.json
```

Results:

- py_compile passed
- high-cap projection regression passed in 208.102s
- combined focused suite passed 7 tests in 391.527s
- projection smoke wrote `/tmp/worldfoam_factorized_i16_projection_smoke.json`
  with `status=ok`
- smoke row: current schema storage `40,048` bytes, projected int16 metadata
  storage `35,946` bytes, savings `4,102` bytes
- the smoke was still timing-contended by unrelated `ai_trader`/pytest CPU work,
  so it is correctness/storage evidence only

## Follow-up: int16 metadata is now the actual factorized layout

I replaced the projection-only factorized metadata path with actual int16
metadata buffers in the Metal hot path. The existing
`owner-run-delta-packed-factorized-recompute-fused-mse-nomid` mode now moves
`base_offsets_i16`, `track_change_offsets_i16`, `change_frame_i16`, and
`change_offsets_i16` to MPS, removes the old int32 metadata keys from the
selected tape, and the factorized Metal wrapper/kernel now require int16
metadata for those four arrays.

Validation:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_highcap_storage_removes_dense_coeff16 -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --frame-counts 2 --render-size 16 --site-count 4 --near 0.0 --far 3.5 --density 8.0 --invalid-epsilon 1.0e-7 --transmittance-threshold 1.0e-4 --steps 1 --warmup-steps 0 --optimizer-mode manual-vjp --tape-mode owner-run-delta-packed-factorized-recompute-fused-mse-nomid --endpoint-record-source slow-owner-run --out-json /tmp/worldfoam_factorized_i16_actual_smoke.json
```

Results:

- native variant rebuilt successfully
- high-cap actual-int16 storage regression passed in 177.899s
- moving-ray Metal loss/site-gradient parity passed in 10.895s
- combined focused suite passed 7 tests in 265.379s
- actual-layout smoke wrote `/tmp/worldfoam_factorized_i16_actual_smoke.json`
  with `status=ok`
- smoke row: schema storage `35,946` bytes, non-coeff MPS resident storage
  `11,306` bytes, int16 metadata keys present, old int32 metadata keys absent
- smoke timing is still contaminated by unrelated `ai_trader`/pytest/Metal
  compiler activity, so this is correctness/storage evidence only

## Follow-up: frame-select factorized shader fork

The first follow-up probe rejected the simple "drop `change_frame_i16` and
infer sparse event frames" idea. Actual moving-camera owner-run tapes are not
per-track consecutive: at render16/site8, `4f` had tracks with change frames
like `[2, 3]` instead of `[1, 2, 3]`, and `8f` had tracks like `[4, 5, 6]`.

I pivoted to a real frame-select fork:

- new tape mode:
  `owner-run-delta-packed-factorized-frameselect-recompute-fused-mse-nomid`
- new selected metadata:
  `frame_change_index_i16`, one selected sparse-change index per
  `(track, frame>0)`; `-1` means use the base row
- removed from the selected resident path:
  `track_change_offsets_i16`, `track_chunk_change_offsets_i16`,
  `change_frame_i16`
- kept:
  sparse `change_offsets_i16` plus packed base/change records and factorized
  `boundary_f32 + track_ray_coeff_f32`
- new native op/kernel:
  `endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only`
  and
  `wf2_endpoint_record_delta_replace_factorized_frameselect_recompute_mse_vjp_direct_atomic_rgb_only_tensor`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/train_eval_owner_run_tape.py research_experiments/world_foam_lane2/test_train_eval_owner_run_delta_packed.py third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_delta_packed_recompute_nomid_matches_owner_run_fused_mse_on_moving_rays -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_frameselect_removes_sparse_frame_scan_metadata -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_owner_run_coeff_factorization research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed research_experiments.world_foam_lane2.test_probe_owner_run_boundary_tape -v
rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --frame-counts 4 --render-size 16 --site-count 8 --near 0.0 --far 3.5 --density 8.0 --invalid-epsilon 1.0e-7 --transmittance-threshold 1.0e-4 --steps 1 --warmup-steps 0 --optimizer-mode manual-vjp --tape-mode owner-run-delta-packed-factorized-frameselect-recompute-fused-mse-nomid --endpoint-record-source slow-owner-run --out-json /tmp/worldfoam_factorized_frameselect_smoke.json
```

Results:

- py_compile passed
- native variant rebuilt successfully
- moving-ray Metal loss/site-gradient parity passed in 14.327s
- 24-site/8f frame-select storage regression passed in 115.271s
- combined focused suite passed 8 tests in 360.265s
- frame-select smoke wrote `/tmp/worldfoam_factorized_frameselect_smoke.json`
  with `status=ok`
- smoke row: schema storage `46,088` bytes, `frame_select_i16=3,072`
  bytes, no `track_change_offsets_i16`/`change_frame_i16` schema keys,
  non-coeff MPS resident storage `21,016` bytes
- smoke timing is still contaminated by unrelated `ai_trader` Python and
  Metal compiler activity, so this is correctness/storage evidence only

## Follow-up: comparison gate and blocked clean timing

I tried to run the clean regular factorized timing ladder with
`--require-benchmark-environment-ok`:

```text
research_experiments/world_foam_lane2/results/2026-05-19_owner_run_delta_packed_factorized_recompute_nomid_ladder_2_4_8_16_render16_site8_clean_compare.json
```

The command started from a clean preflight but ended with
`benchmark_environment.status=contended` after unrelated `pytest`, STAR UVT
training, `ai_trader` Python, and `MTLCompilerService` work appeared. The row
artifact is therefore not clean promotion evidence. The contaminated regular
row snapshot was:

| frames | total ms | backward ms | schema bytes | topology bytes | noncoeff resident | train PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 2.638 | 2.206 | 38,342 | 13,206 | 13,262 | 13.374 |
| 4 | 2.687 | 2.300 | 48,646 | 23,510 | 23,574 | 13.407 |
| 8 | 4.936 | 4.177 | 59,750 | 34,614 | 34,694 | 13.336 |
| 16 | 6.195 | 5.447 | 67,014 | 41,878 | 41,990 | 13.525 |

I added a focused comparison gate:

```text
research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py
research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
```

The gate runs regular factorized and frame-select factorized with per-mode
stable preflight checks, refuses to continue after a contaminated artifact by
default, computes per-frame total/backward/storage ratios, and writes a summary
recommendation (`frameselect_candidate`, `keep_regular_or_fork_again`, or
`rerun_clean`). Fast validation passed:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
```

Result: 4 tests passed.

Running the new gate live wrote:

```text
research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_blocked_live.summary.json
```

It correctly stopped before training with
`status=preflight_failed_before_regular`; top blockers were `python -m pytest
tests/` at `94.7%` CPU and an `ai_trader` Toto residual live quote shadow
process at `79.5%` CPU. Next timing should be:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py \
  --run-id 2026-05-19_factorized_frameselect_compare_clean_site8 \
  --wait-for-benchmark-environment-ok \
  --stable-preflight-checks 2
```

Only if that summary is clean and says `frameselect_candidate` should the same
gate be repeated at site24/high-cap before any STAR UVT competitiveness claim.

## Follow-up: site8 comparison attempt and retry gate

I ran the comparison gate when the preflight turned clean:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py \
  --run-id 2026-05-19_factorized_frameselect_compare_clean_site8_attempt1 \
  --stable-preflight-checks 2 \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 600 \
  --wait-interval-s 15
```

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_clean_site8_attempt1.factorized_frameselect_compare_summary.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_clean_site8_attempt1.regular_factorized.json
research_experiments/world_foam_lane2/results/2026-05-19_factorized_frameselect_compare_clean_site8_attempt1.frameselect_factorized.json
```

The regular artifact is clean (`benchmark_environment.status=background`).
The frame-select artifact is functionally OK but not promotable because its end
snapshot became `contended` after a separate pytest/capped-replay process
appeared. The gate therefore exited nonzero with
`status=frameselect_artifact_contaminated`.

Observed rows:

| frames | regular total ms | frame-select total ms | regular backward ms | frame-select backward ms | regular schema | frame-select schema |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 5.655 | 1.779 | 3.806 | 1.516 | 38,342 | 35,268 |
| 4 | 2.984 | 2.158 | 2.532 | 1.798 | 48,646 | 46,088 |
| 8 | 2.790 | 2.079 | 2.358 | 1.746 | 59,750 | 59,666 |
| 16 | 3.573 | 3.541 | 3.223 | 3.007 | 67,014 | 74,046 |

Interpretation:

- frame-select is directionally faster on the contaminated attempt, especially
  at `2/4/8f`
- the 16f frame-select row nearly ties compute but has worse schema/topology
  storage because the dense `track_count * (frame_count - 1)` frame table
  overtakes the removed sparse scan metadata
- this is not clean promotion evidence because the frame-select artifact ended
  contended
- the next shader idea is a compact per-track frame bitmask plus rank/popcount
  lookup: direct per-frame selection without the dense table

I then extended the gate with `--max-comparison-attempts`, so contaminated
regular/frame-select artifacts can retry automatically with distinct attempt
output files. Fast validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py research_experiments/world_foam_lane2/test_compare_factorized_frameselect_gate.py
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_compare_factorized_frameselect_gate -v
```

Result: 5 tests passed.

The next quiet-window command should include retries:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/compare_factorized_frameselect_gate.py \
  --run-id 2026-05-19_factorized_frameselect_compare_clean_site8_retry \
  --stable-preflight-checks 2 \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 1800 \
  --wait-interval-s 15 \
  --max-comparison-attempts 3
```
