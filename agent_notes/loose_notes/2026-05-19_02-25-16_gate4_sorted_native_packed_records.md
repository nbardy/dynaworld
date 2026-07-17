# Gate4 Sorted-Native Packed Records

## Context

After the Gate4 endpoint-record reflection, live timing promotion was still
blocked by unrelated high-CPU `ai_trader` pytest/RL jobs. I avoided another
timing ladder and took a parity-testable compiler step instead.

The previous native sorted Gate4 delta builder could produce unpacked
owner/left/right tensors, but the packed framegroup16 fused-MSE path still
could not request sorted-native plus emitted packed records together. In
practice that meant the CLI flags could not express the most direct
sorted-native packed-record variant.

## Change

- Added a separate `gate4_delta_replace_packed_from_sorted_cpu` op in
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
  that wraps the existing sorted delta builder and emits `base_record_i32` /
  `change_record_i32` alongside the unpacked tensors. Keeping the old
  unpacked sorted op unchanged preserves its wider owner/cut-id contract; the
  packed constraints now apply only when the explicit packed op is requested.
- Updated `research_experiments/world_foam_lane2/gate4_affine_slab_tape.py`
  so `_append_native_sorted_delta_chunk(...)` accepts optional packed-record
  output lists and the sorted-native path is no longer disabled when
  `experimental_native_emitted_pack_records=True`.
- Extended the high-cap Gate4 compiler test to exercise
  `experimental_native_sorted_delta=True` plus
  `experimental_native_emitted_pack_records=True`, comparing both unpacked rows
  and packed bit layout against the existing Python packed reference.

This does not promote sorted-native timing. It only makes the variant honest
and directly testable.

## Validation

```bash
rtk zsh -lc 'cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace'

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_native_endpoint_record_packer_matches_bit_layout \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v
```

Results:

- rebuild passed
- py_compile passed
- focused native packed/sorted parity tests passed `2/2`
- full Gate4 compiler unit passed `8/8`

## Next

When the benchmark environment is clean, run the promotion wrapper first. If
the keeper revalidates, the next timing candidate is the sorted-native packed
record path under the normal reference-artifact verifier, not a single-row spot.

## Follow-up: promotion wrapper flags

I added native-variant pass-through flags to
`research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py` so
the sorted-native packed path can be timed later without hand-built commands:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --wait-for-benchmark-environment-ok \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_packed_promotion_dryrun \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --dry-run
```

Results:

- wrapper py_compile passed
- wrapper unit tests passed `3/3`
- dry-run summary includes both native flags in `train_command`
- live preflight is still contended, so no timing promotion was attempted

## Follow-up: packed-sorted fallback guard

The sorted-native availability check now asks for the actual op needed by the
requested tape shape: packed sorted op when emitted packed records are requested,
unpacked sorted op otherwise. Without that guard, an older extension binary
with only the unpacked sorted op could take the sorted branch, fail to append
packed rows, and skip the cut-prep/native packed fallback.

Regression coverage was added to patch out
`_gate4_delta_replace_packed_from_sorted_cpu_op()` while leaving the older
sorted op visible, then request sorted-native plus emitted packed records and
assert the fallback tape still matches the Python reference and packed bit
layout.

Validation after the fallback guard:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- py_compile passed
- focused fallback/sorted packed Gate4 test passed `1/1`
- full Gate4 compiler unit passed `8/8`
- promotion wrapper unit tests passed `3/3`
- dry-run summary
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_fallbackfix_dryrun.promotion_summary.json`
  forwards `--experimental-native-sorted-delta` and
  `--experimental-native-emitted-pack-records`
- keeper self-verifier on
  `research_experiments/world_foam_lane2/results/2026-05-18_gate4_endpoint_record_nativechunk_directdelta_repeat20_render64_site24_2_4_8_16.json`
  returned `status=ok`, `clean_speedscale_artifact=true`,
  `total_median_scale=1.3136`, `backward_median_scale=1.3643`,
  `storage_scale=1.0399`
- live preflight is still contended by unrelated `ai_trader` pytest/export/
  quote-shadow jobs and a STAR UVT feature kernel, so timing promotion remains
  intentionally blocked

## Follow-up: pure-Python packed-record fallback

The emitted packed-record request now remains correct even if neither the
packed sorted op nor the packed cuts op is available. `_delta_replace_tape_from_lists(...)`
materializes `base_record_i32` / `change_record_i32` from the unpacked
owner/left/right lists when the native append path leaves the requested record
lists empty. Nonempty length mismatches still raise, so a partially written
native result cannot be silently accepted.

`Gate4EndpointDeltaReplaceTape.storage_bytes` now includes optional
`base_record_i32` and `change_record_i32` tensors. The selected-tape storage
path already estimated packed record bytes directly, but the raw delta tape
metric was undercounting emitted packed tensors.

Validation:

```bash
rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler.Gate4MovingRaySlabCompilerTest.test_highcap_single_slab_sorted_rows_match_cut_array_delta_records -v

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- py_compile passed
- focused packed fallback/storage test passed `1/1`
- full Gate4 compiler unit passed `8/8`
- promotion wrapper unit tests passed `3/3`
- dry-run summary
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_pythonfallback_dryrun.promotion_summary.json`
  still forwards the sorted-native emitted-packed flags to the real promotion
  command

## Follow-up: auto-mode effective emitted-pack flag

Auto framegroup mode resolves to the packed i32 shader for frame counts at or
below `64`, but resolves back to the i16x3 shader above that. The CLI-level
`--experimental-native-emitted-pack-records` request is now split from the
effective per-row/tape behavior: Gate4 only builds native-emitted packed i32
records when the resolved mode actually consumes packed i32 records. This keeps
the `64,128` auto-selector path from paying to materialize unused packed i32
records on the `128f` i16x3 row.

The run metadata now reports both requested and effective emitted-pack flags:

- `experimental_native_emitted_pack_records`
- `experimental_native_emitted_pack_records_effective`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_delta_framegroup_i16x3_packed_train_eval.py \
  research_experiments/world_foam_lane2/gate4_affine_slab_tape.py \
  research_experiments/world_foam_lane2/test_gate4_moving_ray_slab_compiler.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval -v

rtk env PYTHONPATH=third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- py_compile passed
- auto-selector compare unit passed `6/6`
- full Gate4 compiler unit passed `8/8`
- promotion wrapper unit tests passed `3/3`
- dry-run summary
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_auto_effective_dryrun.promotion_summary.json`
  forwards the requested sorted/emitted flags for a future real `64,128`
  promotion run; row-level execution will decide the effective emitted-pack
  behavior after auto resolution
- live timing is still blocked by benchmark preflight contamination from
  unrelated `ai_trader` RL/verifier jobs

## Follow-up: native packed verifier and preflight classifier gate

The native packed path is now guarded by an explicit extension verifier before
the promotion wrapper launches any packed/native timing row. The verifier checks
the native op surface for sorted and cut packed-record builders, then exercises
both no-change and changing-owner fixtures. The changing fixtures assert packed
record layout, frame ids, row offsets, and track offsets for sorted-native and
cut-prep-native construction.

The benchmark-environment classifier was also tightened after idle `pytest`
wrapper processes were incorrectly blocking every timing attempt. Low-CPU
`pytest` wrappers are now background-only, but high-CPU `pytest` still blocks,
and low-CPU `torch` / `mps` / `metal` commands still block because those can
contaminate GPU/accelerator timing even when CPU use is small.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only
```

Results:

- benchmark-environment classifier unit passed `7/7`
- direct native packed extension verifier returned `status=ok`
- broader focused WorldFoam gate passed `62/62`
- live benchmark preflight still returned `status=contended`, but now for true
  high-CPU blockers rather than idle `pytest` wrappers:
  `verify_btc15m_activation_bank_integrity.py` at `95.5%`,
  `verify_btc15m_rl_row_aligned_state` at `94.2%`,
  `train_kalshi_btc15m_rl.py` at `93.9%`,
  `build_btc15m_activation_rl_dataset.py` at `93.3%`, and
  `probe_btc15m_tree_oracle_context_feature_frame` at `5.1%`.

No native packed timing promotion was launched from that contended state. The
correct next run is still the sorted-native emitted-packed promotion wrapper
under a clean or merely background benchmark environment:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_native_packed_verified_clean_timing \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

## Follow-up: promotion wrapper reference guard

`run_framegroup16_promotion_gate.py` now validates reference-artifact frame
coverage before launching preflight/train/eval. The default reference artifact
only covers `2,4,8,16`, so a custom frame ladder such as `64,128` now fails
fast unless the caller passes either a matching `--reference-artifact` or the
explicit exploratory escape hatch `--no-reference-artifact`.

This closes a practical footgun from the `64,128` dry-run: previously the train
command would be shaped correctly, but the verifier command would still carry
the default `2,4,8,16` reference and reject only after the expensive run.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_custom_reference_guard_dryrun \
  --frame-counts 64,128 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --dry-run

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_custom_no_reference_guard_dryrun \
  --frame-counts 64,128 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --no-reference-artifact \
  --dry-run
```

Results:

- wrapper py_compile passed
- wrapper unit tests passed `5/5`
- custom `64,128` dry-run with the implicit default reference failed before
  preflight/train with `status=config_failed`
- custom `64,128 --no-reference-artifact` dry-run succeeded and omitted
  `--reference-artifact` from the verifier command
- default `2,4,8,16` dry-run still uses the keeper reference artifact

## Follow-up: reference coverage, not exact equality

The wrapper reference guard now matches the verifier more closely: a reference
artifact may contain extra frame rows as long as it covers every requested
frame. This keeps broad reference artifacts useful for narrower reruns while
still failing fast when requested frames are missing.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_custom_reference_guard_coverage_dryrun \
  --frame-counts 64,128 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --dry-run

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_default_reference_guard_coverage_dryrun \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --dry-run
```

Results:

- wrapper py_compile passed
- wrapper unit tests passed `7/7`
- explicit superset reference fixture was accepted
- explicit missing-frame reference fixture failed before launch
- custom `64,128` dry-run with the default reference still failed fast because
  `64,128` are not covered
- default `2,4,8,16` dry-run still carries the keeper reference
- custom `64,128 --no-reference-artifact` dry-run still succeeds for exploratory
  no-reference verification

## Follow-up: packed runtime smoke and storage-sidecar breakdown

The sorted-native plus native-emitted packed-record path now has a tiny
end-to-end runtime smoke through the actual MPS packed framegroup fused-MSE
shader. This is deliberately not speed evidence because the benchmark
environment was contended by unrelated `ai_trader` pytest/training jobs, but it
does prove the path reaches the shader, produces finite output, nonzero
gradients, a parameter update, and a heldout PSNR row.

The smoke also clarified an initially confusing storage read. The compact delta
tape itself was `68,284` bytes, but the selected fused-MSE path reported
`295,608` bytes because the half-precision endpoint coefficient sidecar was
`270,336` bytes. `train_eval_owner_run_tape.py` now records
`train_endpoint_record_coeff_storage_bytes` and
`train_endpoint_record_coeff_storage_vs_selected` so future timing artifacts can
separate topology/delta-tape storage from the coefficient sidecar.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_storagebreakdown_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_storagebreakdown_render16_2.partial.json

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- runtime smoke status `ok`
- resolved mode `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- `experimental_native_emitted_pack_records_effective=true`
- gradients nonzero, parameters updated, finite outputs, loss decreased
- heldout PSNR row present (`14.5856` on the tiny 2f/16px smoke)
- final focused unit pass `35/35`
- no real timing promotion was run because preflight remains contended

## Follow-up: precision-aware selected-storage telemetry

The first storage-sidecar field was too coarse: it reported fp16 coefficient
bytes unconditionally, even though some older block-coeff modes store fp32
coefficient sidecars. `train_eval_owner_run_tape.py` now computes selected
coefficient bytes through a mode-aware helper and also reports
`train_selected_tape_topology_storage_bytes` /
`train_selected_tape_topology_storage_vs_full`.

The packed sorted-native runtime smoke was rerun as:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_storagebreakdown_v2_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_storagebreakdown_v2_render16_2.partial.json
```

The v2 artifact reports:

- `train_selected_tape_topology_storage_bytes=25,272`
- `train_endpoint_record_coeff_storage_bytes=270,336`
- `train_selected_tape_storage_bytes=295,608`
- `train_selected_tape_topology_storage_vs_full=0.08549`
- `train_endpoint_record_coeff_storage_vs_selected=0.91451`

Focused validation after the telemetry fix:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Result: `36/36` passed. The live benchmark preflight remained contended by
unrelated `ai_trader` pytest/training jobs plus a STAR UVT feature-kernel
process, so no real promotion timing was run.

## Follow-up: verifier carries topology/coeff storage scales

`verify_framegroup16_timing_robust.py` now preserves the new selected-storage
breakdown in verifier outputs instead of flattening everything into one storage
scale:

- `topology_storage_bytes` per row
- `coeff_storage_bytes` per row
- `topology_storage_scale`
- `coeff_storage_scale`

The verifier keeps old artifacts compatible: the split fields are optional, but
when present they are validated as nonnegative ints and checked against
`--max-topology-storage-scale` / `--max-coeff-storage-scale` (both default
`1.10`). A new regression test makes sure topology storage can fail a promotion
even when total selected storage is still under the broad threshold.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v
```

Result: `46/46` passed. The current live preflight is still contended by
unrelated `ai_trader` RL/model-bank jobs and a STAR UVT feature-kernel process,
so timing promotion remains intentionally blocked.

## Follow-up: promotion summary keeps storage split

`run_framegroup16_promotion_gate.py` now carries the verifier's storage split
into the compact `verify_result` embedded in promotion summaries. Before this,
the verifier JSON had `storage_scale`, `topology_storage_scale`, and
`coeff_storage_scale`, but the promotion wrapper's brief dropped them. A future
real promotion summary will now expose:

- top-level `storage_scale`
- top-level `topology_storage_scale`
- top-level `coeff_storage_scale`
- per-frame `storage_bytes`
- per-frame `topology_storage_bytes`
- per-frame `coeff_storage_bytes`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v
```

Results:

- wrapper/verifier focused tests passed `17/17`
- broader focused shader/promotion gate passed `46/46`
- live preflight still reports unrelated high-CPU pytest jobs, so no promotion
  timing was run

## Follow-up: zero-size storage sidecar verifier edge

The split-storage verifier had one hygiene bug after adding topology/coeff
scales: if an optional sidecar was present but zero bytes across the whole
sweep, the scale calculation divided by zero and produced `inf`, which looked
like a regression. That is wrong for legitimate zero-sized optional storage.

`verify_framegroup16_timing_robust.py` now handles optional split-storage
scales through a helper:

- missing optional split fields still produce `None`
- zero across the sweep reports scale `0.0`
- growth from zero to positive still reports `inf` and fails the threshold

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v
```

Result: `12/12` passed.

Broader focused validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v
```

Result: `48/48` passed.

Live promotion timing remains blocked. The preflight command:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only
```

returned `status=contended` with unrelated `ai_trader` processes near 100%
CPU, including `scripts.verify_btc15m_rl_row_aligned_state` and
`scripts/evaluate_kalshi_btc15m_rl.py --help`. Do not promote a timing result
until this preflight is clean.

## Follow-up: bounded promotion wait still preflight-blocked

Ran the actual promotion wrapper with the native sorted + native emitted packed
record path and a bounded preflight wait:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_native_sorted_packed_wait90 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 90 \
  --wait-interval-s 30
```

Result: `preflight_failed`; no train/eval timing was launched.

Summary artifact:
`research_experiments/world_foam_lane2/results/2026-05-19_native_sorted_packed_wait90.promotion_summary.json`

The wrapper made 4 preflight attempts. The top blocker changed each time but
remained unrelated high-CPU `ai_trader` work:

- attempt 1: `75.3% (Python)`
- attempt 2: `90.7% python -m pytest tests/`
- attempt 3: `96.0% scripts/train_kalshi_btc15m_dqn.py ...`
- attempt 4: `95.6% scripts/audit_btc15m_model_bank_goal_state.py --format json --fail-on-incomplete`

The final preflight snapshot also had several other high-CPU blockers,
including `scripts/audit_btc15m_task_queue_data_gate.py --fail-on-blocked`,
`scripts/evaluate_kalshi_btc15m_rl.py --help`, and the shared feed daemon.

Current conclusion is unchanged but now backed by the promotion wrapper's own
summary: the shader/test lane is ready for promotion, but real speed evidence
is still blocked by host contention. Do not infer sublinear runtime from this
attempt.

## Follow-up: explicit native packed extension verifier

The benchmark preflight is still contended, so the next useful non-timing gate
was proving the compiled fork is actually wired rather than only exercising the
Python fallback. Added:

`research_experiments/world_foam_lane2/verify_native_packed_extension.py`

The verifier imports
`third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0`,
requires these custom ops, and runs a tiny CPU fixture:

- `gate4_delta_replace_from_sorted_cpu`
- `gate4_delta_replace_packed_from_sorted_cpu`
- `pack_endpoint_records_i32_cpu`

It checks that the packed sorted-native op returns the same unpacked topology
as the ordinary sorted-native op, and that `base_record_i32` /
`change_record_i32` match the record packer.

Validation:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result:

```json
{
  "base_offsets_i32": [0, 2],
  "base_record_i32": [2097152, 1049089],
  "change_record_i32": [],
  "status": "ok",
  "track_change_offsets_i32": [0, 0]
}
```

After adding the verifier, the broader focused gate still passes:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v
```

Result: `48/48` passed.

Live benchmark preflight is still blocked by unrelated high-CPU `ai_trader`
work (`python -m pytest tests/`, `scripts/train_kalshi_btc15m_dqn.py`, and
the shared feed daemon were above threshold in the latest snapshot). Timing
promotion remains intentionally unrun.

## Follow-up: promotion wrapper now enforces native packed extension gate

The native packed verifier is now part of
`run_framegroup16_promotion_gate.py` for the specific path it proves:
`--experimental-native-sorted-delta` plus
`--experimental-native-emitted-pack-records`. The wrapper records
`native_packed_extension_verify_command`,
`native_packed_extension_verify_status`, and the verifier JSON result in the
promotion summary. If the verifier fails, the wrapper stops before benchmark
preflight and before train/eval timing.

Unit coverage added in `test_framegroup16_promotion_gate.py`:

- sorted-native emitted-pack dry-runs include the verifier command and status
- verifier failure returns `native_packed_extension_verify_failed`
- verifier failure does not call benchmark preflight

Focused wrapper validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Result: `9/9` passed.

Native verifier validation:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result: `status=ok`.

Broader focused validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v
```

Result: `50/50` passed.

Real wrapper path check:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_native_sorted_packed_verify_gate_preflight \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

Result: native verifier passed, then benchmark preflight failed before timing.
Summary artifact:
`research_experiments/world_foam_lane2/results/2026-05-19_native_sorted_packed_verify_gate_preflight.promotion_summary.json`

The summary recorded:

- `native_packed_extension_verify_status=0`
- verifier result `status=ok`
- `preflight_status=2`
- final status `preflight_failed`

The latest top preflight blockers were unrelated `ai_trader` jobs:
`scripts.verify_btc15m_rl_row_aligned_state` around `97.5%` / `96.8%` CPU and
`scripts/train_kalshi_btc15m_rl.py` around `92.6%` CPU. Timing promotion is
still intentionally unrun.

## Follow-up: native verifier is now in the focused unittest gate

Added `test_verify_native_packed_extension.py` so the native sorted-packed
extension verifier is also covered by the focused unittest suite. The test
skips only if the local extension `.so` is absent; on this built fork it runs
`verify_native_packed_extension.verify()` and asserts the expected packed
fixture output:

- `base_offsets_i32=[0, 2]`
- `base_record_i32=[2097152, 1049089]`
- `change_record_i32=[]`
- `track_change_offsets_i32=[0, 0]`

Focused native-verifier test:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `1/1` passed.

Updated broader focused gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `51/51` passed.

Latest preflight remains contended, so timing promotion was not run. The top
blockers were unrelated `ai_trader` jobs: `scripts/train_kalshi_btc15m_rl.py`
around `99.1%` CPU and `python -m pytest tests/` around `97.3%` CPU.

## Follow-up: native verifier now covers cut-array packed records too

Extended `verify_native_packed_extension.py` to cover both native packed
compiler entry points:

- sorted-row path: `gate4_delta_replace_packed_from_sorted_cpu`
- cut-array path: `gate4_delta_replace_packed_from_cuts_cpu`

The cut-array fixture mirrors the sorted fixture with two frames and no
changes. It asserts the same packed bit layout:

- `cut_base_offsets_i32=[0, 2]`
- `cut_base_record_i32=[2097152, 1049089]`
- `cut_change_record_i32=[]`
- `cut_track_change_offsets_i32=[0, 0]`

Direct verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result: `status=ok`, with both sorted and cut packed fields present.

Focused unittest:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `1/1` passed.

Broader focused gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `51/51` passed.

Benchmark preflight was checked again and still returned `status=contended`.
Top blockers were unrelated `ai_trader` / pytest work:

- `scripts.verify_btc15m_rl_row_aligned_state` around `96.5%` CPU
- `python -m pytest tests/` around `95.7%` CPU
- `shared_feed_daemon.py` around `5.1%` CPU

No timing promotion was run from this state.

## Follow-up: native verifier now catches non-empty change records

The previous native verifier fixtures proved packed base-record layout and
empty change sidecars, but they would not catch a bug where native packed
change records were omitted or packed incorrectly. Extended
`verify_native_packed_extension.py` with changing fixtures for both native
entry points:

- sorted-row path changes owner on frame 1 by setting `site_t=[0.0, 1.0]`
- cut-array path changes owner on frame 1 by setting `initial_owner=[0, 1]`

Both paths now assert:

- `changing_*_track_change_offsets_i32=[0, 1]`
- `changing_*_change_frame_i32=[1]`
- `changing_*_change_offsets_i32=[0, 2]`
- `changing_*_change_record_i32=[2097153, 1049088]`

Direct verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result: `status=ok`; output included non-empty sorted and cut change records.

Focused unittest:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `1/1` passed.

Broader focused gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `51/51` passed.

Benchmark preflight stayed contended, so no timing promotion was run. The latest
blocking processes were unrelated `ai_trader` jobs:

- `lean_trade.runners.run_btc_15m_tree_residual_live_quote_shadow_paper`
  around `69.7%` CPU
- `shared_feed_daemon.py` around `6.0%` CPU

## Follow-up: promotion wrapper now asserts native change-record evidence

The direct verifier already emitted non-empty sorted and cut packed change
records, but the promotion-wrapper test only checked that the native verifier
ran and exited `0`. Tightened
`test_sorted_native_emitted_pack_records_verifies_extension_before_launch` so
the wrapper summary must preserve the verifier payload, including:

- `changing_sorted_change_record_i32=[2097153, 1049088]`
- `changing_cut_change_record_i32=[2097153, 1049088]`

This means the promotion gate test now covers the boundary where the native
packed extension verifier becomes benchmark/promotion evidence.

Direct verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result: `status=ok`, with non-empty sorted and cut packed change records.

Targeted tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_sorted_native_emitted_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_native_extension_verify_failure_stops_before_preflight -v
```

Result: `3/3` passed.

Broader focused gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `51/51` passed.

Benchmark preflight was checked before this patch and remained contended. Top
blockers were unrelated `ai_trader` / pytest work:

- `report_btc15m_strict_reopen_candidate_gaps.py` around `96.7%` CPU
- `build_btc15m_activation_rl_dataset.py` around `95.8%` CPU
- `python -m pytest tests/` around `95.2%` CPU
- `check_btc15m_sft_runtime_parity.py` around `92.8%` CPU
- `shared_feed_daemon.py` around `17.1%` CPU

No promotion timing was run.

## Follow-up: benchmark classifier ignores idle monitor wrappers

The benchmark preflight was briefly down to old BTC15M wrapper processes at
`0.0%` CPU, but those wrappers were still appearing under
`blocking_processes`. That was a classifier leak: low-CPU monitor/screen/login
wrappers can contain hard tokens like `mps` in run ids or config strings even
though they are not doing benchmark-relevant work. The actual child process
will still block if it is high CPU.

`train_eval_owner_run_tape.py` now classifies known low-CPU monitor wrappers as
background before applying the low-CPU hard-keyword block:

- `screen -dms`
- `login -pflq`
- `run_btc15m_overnight_shadow_monitor.py`
- `summarize_btc15m_overnight`
- `watch_schema.py --watch`
- `sky.server.server`

High-CPU wrappers still block because the CPU threshold is checked first.

Added regression coverage:

- `test_low_cpu_monitor_wrappers_do_not_block_promotion`
- `test_high_cpu_monitor_wrapper_still_blocks_promotion`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- classifier py_compile passed
- benchmark-environment unit tests passed `9/9`
- broader focused WorldFoam gate passed `71/71`
- live preflight now reports idle wrappers as background; true blockers remain
  only for current high-CPU work:
  `verify_btc15m_rl_row_aligned_state` at `97.9%` and
  `shared_feed_daemon.py` at `9.2%`

No promotion timing was run.

## Follow-up: verifier payload includes shape/device/contiguity

The native packed verifier now exports the full shader-facing tensor contract
for every packed vector the promotion wrapper trusts:

- value list
- dtype
- device
- shape
- contiguity

The verifier already asserts all packed outputs are CPU, `torch.int32`, 1D, and
contiguous before returning `status=ok`; the wrapper now also requires matching
JSON contract fields. This protects the promotion gate from a zero-exit verifier
payload that has correct values but wrong shape, device, or memory layout.

Added wrapper regressions:

- `test_native_verify_zero_status_bad_shape_stops_before_preflight`
- `test_native_verify_zero_status_bad_device_stops_before_preflight`
- `test_native_verify_zero_status_bad_contiguity_stops_before_preflight`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- py_compile passed
- targeted native verifier plus promotion-wrapper tests passed `21/21`
- direct native packed extension verifier returned `status=ok` with CPU,
  `int32`, 1D shape, contiguous metadata for every trusted packed vector
- broader focused WorldFoam gate passed `69/69`

Timing preflight was checked first and still returned `status=contended`.
Latest blockers:

- `train_kalshi_btc15m_imitation.py` around `94.8%` CPU
- a Python 3.11 stdin process around `79.2%` CPU
- long-running `btc15m_toto_allfold_policy_overnight` wrapper processes still
  present

No promotion timing was run.

## Follow-up: native verifier asserts int32 tensor contract

The native packed extension verifier now checks the shader-facing tensor
contract directly before the promotion wrapper accepts a packed native path.
Every packed output tensor from the sorted and cut fixtures must be:

- a CPU tensor
- `torch.int32`
- 1D
- contiguous

The verifier JSON now also emits dtype markers for the packed records and
offset tensors so the promotion wrapper can reject a zero-exit verifier payload
that claims the right values with the wrong dtype.

New/extended wrapper checks include:

- `base_record_i32_dtype="int32"`
- `change_record_i32_dtype="int32"`
- `base_offsets_i32_dtype="int32"`
- `track_change_offsets_i32_dtype="int32"`
- matching cut-fixture dtype fields
- changing-owner sorted/cut change-record dtype fields

Added regression coverage:

- `test_native_verify_zero_status_bad_dtype_stops_before_preflight`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- py_compile passed
- targeted native verifier plus promotion-wrapper tests passed `18/18`
- direct native packed extension verifier returned `status=ok` with dtype
  markers all equal to `int32`
- broader focused WorldFoam gate passed `66/66`

Timing preflight was checked first and still returned `status=contended`.
Latest high-CPU blockers were:

- `build_btc15m_activation_rl_dataset.py` around `95.4%` CPU
- `verify_btc15m_rl_row_aligned_state` around `94.4%` CPU
- `generate_replay_simulator_golden_journal` one-liner around `92.5%` CPU
- `lean_trade.runners.btc_15m_sft_shadow` around `45.0%` CPU
- `shared_feed_daemon.py` around `6.7%` CPU

No promotion timing was run.

## Follow-up: cut-prep native emitted path also requires verifier

Found one remaining promotion-wrapper gap: the native packed verifier was
required for `--experimental-native-sorted-delta --experimental-native-emitted-pack-records`,
but not for `--experimental-native-cut-prep-delta --experimental-native-emitted-pack-records`.
The Gate4 tape can use `gate4_delta_replace_packed_from_cuts_cpu` in that mode,
so the same extension verifier should guard it before timing/promotion.

Changed `_requires_native_packed_extension_verify(...)` so it now requires the
verifier when emitted packed records are requested with either native sorted
delta or native cut-prep delta. Added
`test_cutprep_native_emitted_pack_records_verifies_extension_before_launch`,
which asserts the wrapper preserves the verifier payload including:

- `changing_cut_change_record_i32=[2097153, 1049088]`

Direct verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result: `status=ok`.

Targeted tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_sorted_native_emitted_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_cutprep_native_emitted_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_native_extension_verify_failure_stops_before_preflight -v
```

Result: `3/3` passed.

Broader focused gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `52/52` passed.

Benchmark preflight still returned `status=contended`; no timing promotion was
run. Current notable blockers were unrelated `ai_trader`/pytest work and a
separate STAR UVT train process:

- `sweep_kalshi_btc15m_threshold_policy.py` around `90.0%` CPU
- `train_kalshi_btc15m_imitation.py` around `88.2%` CPU
- `train_kalshi_btc15m_rl.py` around `65.7%` CPU
- `src/train/train.py ... star_uvt_feature_testvideo...` around `37.6%` CPU

## Follow-up: emitted packed records always require the native verifier

Re-checked the Gate4 tape construction and found one more verifier trigger
gap. Even without `--experimental-native-sorted-delta` or
`--experimental-native-cut-prep-delta`, requesting
`--experimental-native-emitted-pack-records` can still use
`gate4_delta_replace_packed_from_cuts_cpu` when the extension is present. That
means the promotion wrapper should guard every emitted-pack path, not only the
explicit sorted/cut-prep variants.

Changed `_requires_native_packed_extension_verify(...)` to return true for any
`experimental_native_emitted_pack_records` promotion. Added
`test_emitted_pack_records_verifies_extension_before_launch`, which asserts the
wrapper preserves the verifier payload for the emitted-only path, including:

- `changing_cut_change_record_i32=[2097153, 1049088]`

Direct verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result: `status=ok`.

Targeted tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_emitted_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_sorted_native_emitted_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_cutprep_native_emitted_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_native_extension_verify_failure_stops_before_preflight -v
```

Result: `4/4` passed.

Broader focused gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `53/53` passed.

Benchmark preflight remained contended. Latest blockers were unrelated
`ai_trader`/pytest work:

- `generate_replay_simulator_golden_journal` one-liner around `95.8%` CPU
- `verify_btc15m_model_bank_prediction_export.py` around `95.5%` CPU
- `verify_btc15m_activation_bank_integrity.py` around `76.6%` CPU
- `shared_feed_daemon.py` around `5.8%` CPU

No promotion timing was run.

## Follow-up: native packer path also requires the native verifier

Found another promotion-wrapper gap on the non-emitted native packer path.
`--experimental-native-pack-records` uses `pack_endpoint_records_i32_cpu`
inside `train_eval_owner_run_tape.py`, but the promotion wrapper only required
`verify_native_packed_extension.py` for emitted packed records. The direct
verifier already exercises `pack_endpoint_records_i32_cpu`, so promotion should
guard this route too.

Changed `_requires_native_packed_extension_verify(...)` to require the verifier
when either `experimental_native_pack_records` or
`experimental_native_emitted_pack_records` is requested.

Added wrapper tests:

- `test_native_pack_records_verifies_extension_before_launch`
- `test_native_pack_records_verify_failure_stops_before_preflight`

The success test asserts the summary preserves verifier output including:

- `changing_sorted_change_record_i32=[2097153, 1049088]`

Direct verifier:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Result: `status=ok`.

Targeted tests:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_native_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_native_pack_records_verify_failure_stops_before_preflight \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_emitted_pack_records_verifies_extension_before_launch \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate.Framegroup16PromotionGateTests.test_native_extension_verify_failure_stops_before_preflight -v
```

Result: `4/4` passed.

Broader focused gate:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v
```

Result: `55/55` passed.

Benchmark preflight was checked first and still returned `status=contended`.
This snapshot did not show high-CPU blockers, but it still reported long-running
`ai_trader`/pytest wrapper processes as contending:

- `SCREEN ... btc15m_toto_allfold_policy_overnight...`
- `uv run python scripts/run_btc15m_overnight_shadow_monitor.py ...`

No promotion timing was run.

## Follow-up: semantic verifier payload guard

The promotion wrapper no longer trusts only the native verifier process exit
code. Before launching any native packed timing row, it now checks that the
verifier JSON has `status="ok"` and the exact expected packed-record fixture
values for both sorted-native and cut-prep-native changing-owner cases. This
closes the last obvious wrapper footgun where a zero-exit but malformed or
semantically failed verifier payload could still allow a packed shader timing
run to start.

Added tests:

- `test_native_verify_zero_status_bad_payload_stops_before_preflight`
- `test_native_verify_zero_status_without_json_stops_before_preflight`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_native_packed_semantic_guard_dryrun \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --dry-run
```

Results:

- wrapper py_compile passed
- promotion-wrapper unit tests passed `15/15`
- broader focused WorldFoam gate passed `64/64`
- direct native packed extension verifier returned `status=ok`
- dry-run summary
  `research_experiments/world_foam_lane2/results/2026-05-19_native_packed_semantic_guard_dryrun.promotion_summary.json`
  carries the native verifier command and sorted/emitted native flags

Live preflight was checked again before this non-timing guard work and still
returned `status=contended`, now with true high-CPU blockers:

- `show_btc15m_next_source_inputs.py` around `94.3%` CPU
- `ai_trader_gpt55_alpha_run2/.../verify_btc15m_activation_bank_integrity.py`
  around `92.3%` CPU
- `python -m pytest tests/` around `43.8%` CPU

No promotion timing was run from that contaminated state.

## Follow-up: semantic verifier also covers no-change offsets

The promotion wrapper's semantic native-verifier guard now checks the no-change
row-offset and track-offset fields as well as the packed owner/left/right
records. These fields are part of the shader-facing packed-record contract:
a zero-exit verifier with correct record words but broken offsets would still
misaddress base/change rows in the timing path.

Added to the required verifier payload:

- `base_offsets_i32=[0, 2]`
- `track_change_offsets_i32=[0, 0]`
- `cut_base_offsets_i32=[0, 2]`
- `cut_track_change_offsets_i32=[0, 0]`

Added regression coverage:

- `test_native_verify_zero_status_bad_offsets_stops_before_preflight`

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- wrapper py_compile passed
- promotion-wrapper unit tests passed `16/16`
- broader focused WorldFoam gate passed `65/65`

Timing preflight was checked first and still returned `status=contended`:

- `train_kalshi_btc15m_imitation.py` around `86.8%` CPU
- `shared_feed_daemon.py` around `6.7%` CPU
- long-running `btc15m_toto_allfold_policy_overnight` wrapper processes still
  present

No promotion timing was run.

## Follow-up: train path validates packed-record tensor contract

The promotion wrapper and native verifier were already strict, but the actual
train path still accepted selected `base_record_i32` / `change_record_i32`
tensors and moved them to MPS without re-checking the shader-facing tensor
contract at the consumption boundary. `train_eval_owner_run_tape.py` now
validates packed endpoint records before MPS transfer:

- CPU tensor
- `torch.int32`
- 1D
- exact shape match against the corresponding owner row tensor
- contiguous storage

This guard applies to both native-emitted packed records and locally packed
records, including the native packer path. It fails before constructing the MPS
shader payload if a bad packed tensor reaches the fused-MSE branch.

Added focused regression coverage in
`test_compare_endpoint_run_record_edit_train_eval.py` for valid records and bad
dtype, shape, contiguity, and device.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_contractguard_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_contractguard_render16_2.partial.json

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only
```

Results:

- py_compile passed
- endpoint compare unit passed `20/20`
- broader focused WorldFoam gate passed `76/76`
- direct native packed extension verifier returned `status=ok`
- runtime smoke status `ok`; resolved mode
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`;
  `experimental_native_emitted_pack_records_effective=true`; gradients were
  nonzero, parameters updated, outputs finite, loss decreased, and heldout
  PSNR was present (`14.5856` on the tiny 2f/16px smoke)
- live benchmark preflight still returned `status=contended`

Latest timing blockers were unrelated high-CPU jobs:

- `python -m pytest tests/` around `93.9%` CPU
- `scripts.verify_btc15m_rl_row_aligned_state` around `93.9%` / `92.6%` CPU
- `scripts.snapshot_btc15m_gate_status` around `91.2%` CPU
- `shared_feed_daemon.py` around `29.2%` CPU

No promotion timing was run from this contaminated state.

## Follow-up: delta index tables validated before shader payload

The prior train-path guard validated packed base/change record tensors, but the
same packed/delta shader payload still trusted the row-offset and change-frame
tables after tape construction. `train_eval_owner_run_tape.py` now validates
the delta index tables before moving them to MPS:

- `base_offsets_i32`, `track_change_offsets_i32`, `change_frame_i32`, and
  `change_offsets_i32` must be CPU `torch.int32`, 1D, and contiguous
- base offsets and track-change offsets must have matching track counts
- `change_offsets_i32` length must equal `change_frame_i32` length plus one
- all offset vectors must start at zero, be monotonic, and end at the tensor
  count they index

This validation now runs through the normal delta tape move helper, the minimal
packed-delta move helper, and the kernel-order packed branch before the shader
payload is constructed.

Added focused regression coverage in
`test_compare_endpoint_run_record_edit_train_eval.py` for valid index tables,
mismatched track counts, bad change-offset length, nonmonotonic offsets, and
final-offset/tensor-count mismatch.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_tableguard_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_tableguard_render16_2.partial.json
```

Results:

- py_compile passed
- endpoint compare unit passed `25/25`
- broader focused WorldFoam gate passed `81/81`
- direct native packed extension verifier returned `status=ok`
- runtime smoke status `ok`; resolved mode
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`;
  `experimental_native_emitted_pack_records_effective=true`; gradients were
  nonzero, parameters updated, outputs finite, loss decreased, and heldout
  PSNR remained `14.5856`
- runtime smoke benchmark environment was `contended`, so this is functional
  shader-path evidence, not timing evidence

Live timing/promotion remains blocked by unrelated high-CPU jobs, including a
STAR UVT feature train process, `python -m pytest tests/`,
`scripts.verify_btc15m_rl_row_aligned_state`, BTC15M planning/status jobs, and
`shared_feed_daemon.py`. Do not promote speed from this smoke.

## Follow-up: component payload guards before native/Python packing

The prior guards validated already-packed record tensors and delta index tables,
but the record component packers could still silently coerce bad owner/left/right
component tensors before Python or native packing. `train_eval_owner_run_tape.py`
now validates endpoint record components before all i32/i16 packers:

- owner/left/right components must be CPU `torch.int32`, 1D, and contiguous
- owner/left/right component shapes must match
- the native emitted i32 path receives the already-validated component tensors,
  not silently cast copies
- delta index-table validation also checks `base_owner_i32` and
  `change_owner_i32` before using owner tensor lengths for final-offset checks

Added focused regression coverage for bad owner dtype in the delta table guard,
bad component dtype, component shape mismatch, and noncontiguous i16x3 component
packing.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_componentguard_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_componentguard_render16_2.partial.json
```

Results:

- py_compile passed
- native packed extension verifier returned `status=ok`
- broader focused WorldFoam gate passed `85/85`
- runtime smoke status `ok`; resolved mode
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- `experimental_native_emitted_pack_records_effective=true`
- gradients were nonzero, parameters updated, outputs finite, loss decreased,
  and tiny heldout PSNR remained `14.5856`
- benchmark preflight still returned `status=contended`; blockers included
  high-CPU `python -m pytest tests/` and
  `verify_btc15m_activation_bank_integrity.py`

Conclusion: this is now strong functional evidence for the sorted native packed
Framegroup16 shader path and its train-path contract guards. It is not a valid
speed/promotion run.

## Follow-up: semantic range guards and blocked clean promotion

Tried the clean promotion wrapper for the current sorted-native/emitted-packed
path:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_packed_componentguard_promotion \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

The wrapper first verified the native packed extension contract (`status=ok`),
then stopped before train/eval because the live preflight changed from
background-only to `status=contended`. The promotion summary is
`research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_componentguard_promotion.promotion_summary.json`.
Top blockers were:

- `scripts/evaluate_btc15m_fold_decile_concentration_guard.py` at `114.1%`
- `lean_trade.runners.btc_15m_sft_shadow` at `95.3%`

No timing artifact was launched from that contaminated state.

After that, added semantic payload guards before packed endpoint records become
shader inputs:

- endpoint record components can now be checked against the current
  `site_count` and `boundary_count`
- packed int32 endpoint records can now be decoded and checked for nonnegative
  signed storage, owner code `< site_count`, and left/right cut ids
  `< boundary_count`
- the checks are wired into the delta packed framegroup/i16/i16cols/i16x4
  train path, including the native-emitted packed-record branch

This closes a real shader safety gap: dtype/shape/contiguity was not enough to
prevent a malformed packed record from decoding to an out-of-range owner or
boundary id inside Metal.

Validation:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval -v

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_semanticguard_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_semanticguard_render16_2.partial.json
```

Results:

- py_compile passed
- endpoint compare unit passed `34/34`
- native packed extension verifier returned `status=ok`
- broader focused WorldFoam gate passed `90/90`
- runtime smoke status `ok`; resolved mode
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- `experimental_native_emitted_pack_records_effective=true`
- gradients were nonzero, parameters updated, outputs finite, loss decreased,
  and tiny heldout PSNR stayed `14.5856`
- runtime smoke `benchmark_environment.status=contended`, with a high-CPU
  `python -m pytest tests/` visible at end; smoke is functional evidence only

Next benchmark action is still the same: wait for a clean/background-only
preflight, then rerun the wrapper promotion gate. Do not cite the smoke timings.

## Follow-up: native packer rank contract

The live benchmark preflight is still blocked by unrelated high-CPU jobs, so no
promotion timing was launched. Current blockers include multiple `ai_trader`
Python jobs and PyTest children at `~90%+` CPU.

While timing was blocked, tightened the native packed-record helper itself. The
Python train path already required packed endpoint record components to be 1D,
but the compiled `pack_endpoint_records_i32_cpu` op only checked dtype, matching
shape, and contiguity. That meant a stale or direct native caller could pass a
rank-2 contiguous tensor and receive rank-2 packed records, outside the shader
payload contract.

Changes:

- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
  now requires `owner_i32`, `left_i32`, and `right_i32` to be rank-1 in
  `pack_endpoint_records_i32_cpu`
- rebuilt the `world_foam_lane2_fused_slab_v0` extension in place
- `verify_native_packed_extension.py` now asserts the compiled op rejects
  rank-2 records, owner ids outside `[-1,255]`, and cut ids that exceed packed
  12-bit cut-code capacity
- `test_verify_native_packed_extension.py` checks those rejection fields, so a
  stale/permissive native build fails the focused gate

Validation:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_native_rankguard_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_native_rankguard_render16_2.partial.json
```

Results:

- native extension rebuild succeeded
- native packed extension verifier returned `status=ok` and reported:
  - `pack_endpoint_records_i32_rejects_rank2=true`
  - `pack_endpoint_records_i32_rejects_owner_out_of_range=true`
  - `pack_endpoint_records_i32_rejects_cut_out_of_range=true`
- native verifier unit passed `1/1`
- broader focused WorldFoam gate passed `90/90`
- runtime smoke status `ok`; resolved mode
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- `experimental_native_emitted_pack_records_effective=true`
- gradients were nonzero, parameters updated, outputs finite, loss decreased,
  and tiny heldout PSNR stayed `14.5856`
- runtime smoke `benchmark_environment.status=contended`; do not cite timing

This makes the compiled native packer obey the same payload rank contract as the
Python train-path guard and makes the promotion wrapper's native verifier catch
stale/permissive builds before any timing sweep.

## Follow-up: native cut-row activity contract

The live benchmark preflight remains `status=contended`, so promotion timing is
still blocked. The current blockers include high-CPU `ai_trader` Python jobs,
PyTest children, and occasional `MTLCompilerService` activity. No timing sweep
was launched.

While timing was blocked, tightened another native emitted-record input
contract. The native cut-row builders could silently produce empty endpoint rows
when `start_segment` and `initial_owner` were malformed, e.g. an active
`start_segment` paired with inactive owner `-1`, or a `start_segment` equal to
the row's segment count. That can hide upstream cut-prep bugs and make the
packed shader path look valid while dropping row work.

Changes:

- `cut_row_is_active(...)` in
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
  now enforces:
  - `start_segment >= -1`
  - `initial_owner >= -1`
  - `start_segment` and `initial_owner` are both active or both inactive
  - active rows have at least one segment
  - active `start_segment` is strictly less than the row segment count
- wired the helper into both `gate4_delta_replace_from_cuts_cpu` and
  `gate4_delta_replace_packed_from_cuts_cpu`
- `verify_native_packed_extension.py` now verifies both packed and unpacked cut
  builders reject out-of-bounds active starts and active/inactive mismatches
- `test_verify_native_packed_extension.py` asserts those rejection fields

Validation:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_cutrowguard_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_cutrowguard_render16_2.partial.json
```

Results:

- native extension rebuild succeeded
- native packed extension verifier returned `status=ok`
- verifier reports:
  - `gate4_delta_replace_from_cuts_rejects_start_segment_oob=true`
  - `gate4_delta_replace_packed_from_cuts_rejects_start_segment_oob=true`
  - `gate4_delta_replace_from_cuts_rejects_active_mismatch=true`
  - `gate4_delta_replace_packed_from_cuts_rejects_active_mismatch=true`
- native verifier unit passed `1/1`
- broader focused WorldFoam gate passed `90/90`
- runtime smoke status `ok`; resolved mode
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- `experimental_native_emitted_pack_records_effective=true`
- gradients were nonzero, parameters updated, outputs finite, loss decreased,
  and tiny heldout PSNR stayed `14.5856`
- runtime smoke `benchmark_environment.status=contended`; do not cite timing

This closes another native cut-prep failure mode before the sorted/cut emitted
packed records become shader payload. Next benchmark action is unchanged: rerun
the wrapper promotion gate only after `--benchmark-environment-check-only`
returns clean/background-only.

## 2026-05-19 sorted-native semantic input guards

After pausing on timing, I used the contended window to tighten the native
sorted-builder contract that feeds the packed endpoint-record path. The
previous native code checked rank/dtype/shape/contiguity and `valid_counts`
bounds, but it still treated any nonzero `row_active_i64` value as active and
only discovered bad `sorted_ids_i64` values later while walking cut
transitions. That left room for upstream sorted-prep bugs to masquerade as a
valid shader payload.

Changes:

- added `sorted_row_is_active(...)` in
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
  so sorted rows must be binary `0/1`
- added `checked_sorted_boundary_id(...)` and
  `checked_sorted_boundary_id_nonnegative(...)`
- `gate4_cut_arrays_from_sorted_cpu` now rejects negative sorted boundary ids
  before emitting cut arrays
- `gate4_delta_replace_from_sorted_cpu` now rejects sorted boundary ids outside
  `[0, boundary_count)` before emitting endpoint records
- `gate4_delta_replace_packed_from_sorted_cpu` inherits the same checks through
  the unpacked sorted builder
- `verify_native_packed_extension.py` now verifies both unpacked and packed
  sorted builders reject:
  - non-binary `row_active_i64`
  - out-of-bounds `valid_counts_i64`
  - negative sorted boundary ids
  - sorted boundary ids greater than or equal to `boundary_count`
- `test_verify_native_packed_extension.py` asserts those new verifier fields

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py

rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_sortedguards_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_sortedguards_render16_2.partial.json
```

Results:

- Python compile passed
- native extension rebuild succeeded
- native verifier returned `status=ok`
- new verifier fields are true for unpacked and packed sorted builders:
  - `gate4_delta_replace_from_sorted_rejects_row_active_bad_value`
  - `gate4_delta_replace_packed_from_sorted_rejects_row_active_bad_value`
  - `gate4_delta_replace_from_sorted_rejects_valid_count_oob`
  - `gate4_delta_replace_packed_from_sorted_rejects_valid_count_oob`
  - `gate4_delta_replace_from_sorted_rejects_negative_boundary_id`
  - `gate4_delta_replace_packed_from_sorted_rejects_negative_boundary_id`
  - `gate4_delta_replace_from_sorted_rejects_boundary_id_oob`
  - `gate4_delta_replace_packed_from_sorted_rejects_boundary_id_oob`
- native verifier unit passed `1/1`
- broader focused WorldFoam gate passed `90/90`
- tiny MPS train/eval smoke returned `status=ok`
- smoke resolved to
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- `experimental_native_sorted_delta=true`
- `experimental_native_emitted_pack_records_effective=true`
- gradients were nonzero, parameters updated, loss decreased, and tiny
  train/heldout PSNR were `12.3217` / `14.5856`
- smoke `benchmark_environment.status=contended`; the timing fields are useful
  only as a functional run record, not as benchmark evidence

Current blocker:

- promotion timing remains blocked by external load. The latest smoke saw
  contending `ai_trader`/pytest/Metal compiler processes. The next promotion
  action is still to wait for a clean/background-only benchmark preflight, then
  rerun `run_framegroup16_promotion_gate.py` with
  `--experimental-native-sorted-delta --experimental-native-emitted-pack-records`.

Follow-up verifier tightening in the same work chunk:

- `verify_native_packed_extension.py` now treats
  `gate4_cut_arrays_from_sorted_cpu` as a required native op
- the verifier records its good-path cut ids, offsets, start segments, and
  initial owners
- the verifier now also checks the standalone cut-array op rejects:
  - non-binary `row_active_i64`
  - out-of-bounds `valid_counts_i64`
  - negative sorted boundary ids
- `test_verify_native_packed_extension.py` asserts those new cut-array fields
- re-ran Python compile, native verifier, native verifier unit, and the broader
  focused gate after this addition; the focused gate still passed `90/90`

Boundary transition table tightening:

- added `validate_boundary_other_table(...)` in the native fused slab binding
  so every `boundary_other_by_owner_i64` entry must be `-1` or a valid site id
- wired it into:
  - `gate4_delta_replace_from_cuts_cpu`
  - `gate4_delta_replace_packed_from_cuts_cpu`
  - `gate4_delta_replace_from_sorted_cpu`
- this catches bad transition-table entries even when a tiny fixture would not
  traverse the specific owner/boundary pair
- extended `verify_native_packed_extension.py` and
  `test_verify_native_packed_extension.py` with cut, packed-cut, sorted, and
  packed-sorted rejection fields for invalid boundary transition tables
- rebuilt the native extension again
- verifier returned `status=ok`
- native verifier unit passed `1/1`
- broader focused WorldFoam gate passed `90/90`
- final tiny MPS smoke returned `status=ok` at
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_boundaryguards_render16_2.json`
  with native sorted delta enabled, emitted packed records effective, resolved
  packed framegroup16 fused-MSE mode, nonzero gradients, parameter updates, and
  train/heldout PSNR `12.3217` / `14.5856`
- final smoke remained `benchmark_environment.status=contended`; still no
  timing/speed/sublinear promotion claim

Sorted depth semantic guards:

- added `checked_sorted_depth(...)` and `check_sorted_depth_order(...)` in the
  native fused slab binding
- `gate4_cut_arrays_from_sorted_cpu` and `gate4_delta_replace_from_sorted_cpu`
  now reject valid sorted-depth candidates that are:
  - NaN or infinite
  - outside `[near, far]`
  - decreasing within the valid candidate prefix for a row/frame
- `gate4_delta_replace_packed_from_sorted_cpu` inherits the same checks through
  the sorted delta builder
- extended `verify_native_packed_extension.py` with explicit fixtures for:
  - `nan_depth`
  - `below_near_depth`
  - `above_far_depth`
  - `decreasing_depth`
- `test_verify_native_packed_extension.py` now asserts those failures for the
  cut-array, unpacked sorted, and packed sorted paths
- rebuilt the native extension
- verifier returned `status=ok` with the new rejection fields true
- native verifier unit passed `1/1`
- broader focused WorldFoam gate passed `90/90`
- tiny MPS smoke returned `status=ok` at
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_depthguards_render16_2.json`
  with native sorted delta enabled, emitted packed records effective, resolved
  packed framegroup16 fused-MSE mode, nonzero gradients, parameter updates, and
  train/heldout PSNR `12.3217` / `14.5856`
- smoke `benchmark_environment.status=contended`; it remains functional
  evidence only, not timing evidence

Cut-array payload semantic guards:

- added `checked_cut_depth(...)` and `validate_cut_row_arrays(...)` in the
  native fused slab binding
- `gate4_delta_replace_from_cuts_cpu` and
  `gate4_delta_replace_packed_from_cuts_cpu` now reject cut-array payloads
  with:
  - NaN or infinite cut depths
  - decreasing cut depths inside a row
  - missing `-1` near sentinel
  - missing `-2` far sentinel
  - internal cut ids outside the boundary table
  - a one-cut row that cannot contain both sentinels
- `verify_native_packed_extension.py` and
  `test_verify_native_packed_extension.py` assert those failures for both
  unpacked and packed cut builders

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_cutarrayguards_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_cutarrayguards_render16_2.partial.json
```

Results:

- Python compile passed
- native verifier returned `status=ok` with the new cut-array rejection fields
  true
- native verifier unit passed `1/1`
- tiny MPS smoke returned `status=ok` at
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_cutarrayguards_render16_2.json`
- smoke resolved to
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- native sorted delta was enabled and emitted packed records were effective
- gradients were nonzero, parameters updated, loss decreased, and tiny
  train/heldout PSNR stayed `12.3217` / `14.5856`
- smoke `benchmark_environment.status=contended`, with unrelated `ai_trader`
  pytest/training jobs near 95% CPU; timing numbers from this smoke are
  functional telemetry only and should not be used as speed evidence

Current-state focused gate after cut-array payload guards:

Before attempting any new timing promotion I reran the live benchmark preflight:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --benchmark-environment-check-only
```

Result: exit `2`, `status=contended`. Current blockers were unrelated
`ai_trader` jobs near 96-98% CPU:

- `scripts/train_kalshi_btc15m_imitation.py`
- `scripts/train_kalshi_btc15m_rl.py`
- `scripts.verify_btc15m_rl_row_aligned_state`

I did not run the promotion wrapper from that state.

Instead I reran the broad focused WorldFoam unit/regression gate against the
current native fork:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Result: `90/90` passed. This is current correctness evidence for the sorted
native packed-record path, the cut-prep/native emitted-pack guard, the verifier
contract, and the promotion-wrapper preflight behavior. It is not timing
promotion evidence.

## Follow-up: promotion wrapper requires semantic verifier guards

The native verifier unit asserted all of the sorted/cut rejection fields, but
the promotion wrapper still only required the verifier `status=ok` plus packed
payload values/dtypes/shapes. That meant a future verifier result with a false
semantic guard field could still pass wrapper preflight if the process exited
zero.

I tightened `run_framegroup16_promotion_gate.py` so
`REQUIRED_NATIVE_PACKED_VERIFY_VALUES` also requires the boolean rejection
fields for:

- packed endpoint-record rank/owner/cut bounds
- cut-row activity, boundary table, depth, sentinel, and internal-id guards
- sorted-row activity/count/id/table/depth guards
- standalone sorted-to-cut-array activity/count/id/depth guards

`test_framegroup16_promotion_gate.py` now builds mock verifier success payloads
from the wrapper's required-value table and adds
`test_native_verify_zero_status_missing_semantic_guard_stops_before_preflight`.
That test patches a false
`gate4_delta_replace_packed_from_cuts_rejects_nan_depth` field and proves the
wrapper returns `native_packed_extension_verify_failed` without calling
preflight.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit passed `21/21`
- broad focused WorldFoam gate passed `91/91`

I also ran the real promotion wrapper with sorted-native emitted-packed flags:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_semantic_verify_preflight_blocked \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

The wrapper executed the stricter native verifier successfully
(`native_packed_extension_verify_status=0`, no verifier failures), forwarded the
two native flags in the train command, then stopped at preflight with
`status=preflight_failed` / `preflight_status=2`. Summary artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_semantic_verify_preflight_blocked.promotion_summary.json
```

The latest preflight blockers were still unrelated `ai_trader` jobs around
100% / 97.7% CPU plus `shared_feed_daemon.py` at 26.5% CPU, so this remains a
wrapper/contract verification, not timing evidence.

## Follow-up: promotion wrapper requires cut-array good-path payloads

The wrapper required semantic guard booleans for the standalone
`gate4_cut_arrays_from_sorted_cpu` op, but it still did not require the
standalone cut-array good-path tensors themselves. I added the expected
`cut_array_cut_ids_i64`, `cut_array_cut_offsets_i64`,
`cut_array_start_segments_i64`, and `cut_array_initial_owner_i64` payloads to
`REQUIRED_NATIVE_PACKED_VERIFY_VALUES`, with automatic dtype/device/shape/
contiguity checks for both `*_i32` and `*_i64` tensor lists.

`test_framegroup16_promotion_gate.py` now includes
`test_native_verify_zero_status_bad_cut_array_payload_stops_before_preflight`.
That test patches a bad `cut_array_cut_offsets_i64=[0,2,6]` while the verifier
process otherwise exits zero, and proves the wrapper fails before preflight.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit passed `22/22`
- broad focused WorldFoam gate passed `92/92`

I also reran the real promotion wrapper:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_cutarray_payload_verify_preflight_blocked \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

The stricter wrapper accepted the actual verifier JSON, including:

- `cut_array_cut_ids_i64=[-1,0,-2,-1,0,-2]`
- `cut_array_cut_offsets_i64=[0,3,6]`
- `cut_array_start_segments_i64=[0,0]`
- `cut_array_initial_owner_i64=[0,0]`

It then stopped at `status=preflight_failed` / `preflight_status=2` before
launching timing. Summary artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_cutarray_payload_verify_preflight_blocked.promotion_summary.json
```

Latest blockers were down to an `ai_trader` RL training subprocess at 42.4% CPU
and `shared_feed_daemon.py` at 8.7% CPU, still enough for the preflight to
reject timing promotion.

## Follow-up: promotion wrapper requires exact native variant root

The verifier already reports `variant_root`, but the promotion wrapper did not
require it. I added `variant_root=str(VARIANT_ROOT)` to
`REQUIRED_NATIVE_PACKED_VERIFY_VALUES` so a stale or wrong native fork cannot
pass the promotion gate by returning matching small payload tensors.

`test_framegroup16_promotion_gate.py` now includes
`test_native_verify_zero_status_wrong_variant_root_stops_before_preflight`,
which patches the verifier result to `/tmp/wrong_world_foam_variant` and proves
the wrapper fails before preflight.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit passed `23/23`
- broad focused WorldFoam gate passed `93/93`

I also reran the real promotion wrapper:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_variantroot_verify_preflight_blocked \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

The stricter wrapper accepted the actual verifier `variant_root`:

```text
/Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
```

It then stopped at `status=preflight_failed` / `preflight_status=2` before
timing. Summary artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_variantroot_verify_preflight_blocked.promotion_summary.json
```

Latest blockers were still unrelated `ai_trader` verification/evaluation jobs
at 98.8%, 88.8%, and 12.7% CPU.

## Follow-up: clean-promotion attempt reblocked by live pytest

After a standalone benchmark preflight briefly returned `status=background`, I
started the real sorted-native/emitted-packed promotion wrapper:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_emitted_packed_clean_promotion \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

The wrapper's stricter native verifier passed again:

- `native_packed_extension_verify_status=0`
- verifier `status=ok`
- verifier `variant_root` matched
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0`
- packed-record payloads, cut-array payloads, dtype/device/shape/contiguity,
  and all semantic rejection guards matched the required promotion contract

The wrapper then re-ran its own benchmark preflight and stopped before
train/eval with `status=preflight_failed` / `preflight_status=2`. Summary
artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_packed_clean_promotion.promotion_summary.json
```

That summary has no result JSON and no reference verifier JSON because timing
was never launched. The blocking snapshot was a high-CPU unrelated pytest tree:

- `(Python)` at 88.9% CPU, pid `38768`
- `python -m pytest tests/` at 5.8% CPU, pid `37014`

Current conclusion: the native packed extension and promotion guard are much
stronger, and the tiny MPS runtime smoke continues to prove the shader path is
functional. But the current run did not produce a valid speed, PSNR sweep, or
sublinear promotion artifact. Stop treating this as an implementation loop
until a genuinely clean/background preflight window is available.

## Follow-up: stable preflight promotion gate

The previous attempt exposed a race: a standalone preflight could briefly look
background-only, but the wrapper's own preflight could immediately see a
different high-CPU blocker. I added a stable-preflight option to
`run_framegroup16_promotion_gate.py`:

```bash
--stable-preflight-checks N
```

When used with `--wait-for-benchmark-environment-ok`, the wrapper now requires
`N` consecutive successful benchmark-environment checks before train/eval is
allowed to launch. The live summary records `success_streak` per attempt and
uses `waiting_for_stable_preflight` while a clean streak is not long enough.
The option also fails closed if someone requests more than one stable check
without waiting.

Regression coverage added in `test_framegroup16_promotion_gate.py`:

- one test simulates `background -> contended -> background -> background` and
  proves the wrapper waits until the consecutive streak reaches `2`
- one test proves `--stable-preflight-checks 2` without waiting does not accept
  a single successful preflight

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit tests passed `25/25`
- broader focused WorldFoam gate passed `95/95`

Then I ran a bounded real promotion attempt with the stricter stable gate:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_emitted_packed_stable2_promotion \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 120 \
  --wait-interval-s 10 \
  --stable-preflight-checks 2
```

Result: `preflight_failed`; no train/eval timing was launched. Summary artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_packed_stable2_promotion.promotion_summary.json
```

Evidence:

- native packed extension verifier passed (`native_packed_extension_verify_status=0`)
- `stable_preflight_checks=2`
- `wait_for_benchmark_environment_ok=true`
- 11 preflight attempts were made over the bounded wait
- all attempts stayed `status=contended`
- every attempt had `success_streak=0`
- no output JSON or reference verifier JSON was created

Top blockers across attempts were unrelated external jobs, including
`lean_trade.runners.run_btc_15m_tree_residual_replay_paper`,
`scripts.verify_btc15m_rl_row_aligned_state`, `python -m pytest tests/`,
`sweep_kalshi_btc15m_threshold_policy.py`, `train_kalshi_btc15m_dqn.py`,
`evaluate_kalshi_btc15m_rl.py`, `train_kalshi_btc15m_imitation.py --help`,
and finally `shared_feed_daemon.py` at `9.0%` CPU.

Conclusion: the promotion gate is stricter and better aligned with the speed
claim we need, but the sorted-native emitted-packed path still has no fresh
valid speed promotion. The next speed run should use the stable gate and wait
for two consecutive clean/background preflights before launching timing.

## Follow-up: stable preflight failure reason fields

I tightened the stable-preflight summary contract so a blocked promotion run
explains why it did not launch timing without requiring manual attempt parsing.
`run_framegroup16_promotion_gate.py` now records:

- `preflight_required_success_streak`
- `preflight_current_success_streak`
- `preflight_max_success_streak`
- `preflight_failure_reason`

Failure reasons distinguish:

- `stable_preflight_streak_not_reached` when at least one clean/background
  sample was observed but the required consecutive streak was not reached
- `benchmark_environment_never_clean` when every preflight attempt was
  contended
- `benchmark_environment_preflight_failed` for other preflight failures

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit tests passed `27/27`
- broader focused WorldFoam gate passed `97/97`

I then ran a short current-format stable promotion attempt:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_emitted_packed_stable2_reasoncheck \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 45 \
  --wait-interval-s 10 \
  --stable-preflight-checks 2
```

Result: `preflight_failed`; no train/eval timing was launched. Summary artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_packed_stable2_reasoncheck.promotion_summary.json
```

Evidence:

- native packed extension verifier passed (`native_packed_extension_verify_status=0`)
- `preflight_failure_reason=benchmark_environment_never_clean`
- `preflight_required_success_streak=2`
- `preflight_current_success_streak=0`
- `preflight_max_success_streak=0`
- 5 preflight attempts were made
- all 5 attempts were `status=contended`
- no output JSON or reference verifier JSON was created

Top blockers were external processes, including `build_btc15m_activation_rl_dataset.py`,
`python -m pytest tests/`, `evaluate_kalshi_btc15m_rl.py`,
`train_kalshi_btc15m_dqn.py`, a Toto residual replay job, and a STAR UVT
feature train process. This remains a correct stop, not speed evidence.

## Follow-up: stale promotion artifact guard

The current benchmark host remains contended, so I did not launch another
timing sweep. The latest direct benchmark preflight still returned
`status=contended`; blockers included a high-CPU `python -m pytest tests/`
process around `98.5%` CPU and `shared_feed_daemon.py` around `22.2%` CPU.

While timing was blocked, I closed another promotion-evidence footgun:
reusing a `--run-id` could leave an old `out_json`, `partial_out_json`, or
`verify_json` beside a new `preflight_failed` summary. That would make the
directory look like a timing run existed even though the current promotion
wrapper stopped before train/eval.

`run_framegroup16_promotion_gate.py` now checks the output, partial, and
reference-verifier artifact paths before native verification/preflight. By
default, any pre-existing output artifact causes `status=config_failed` before
preflight. The user must choose a new `--run-id`, remove stale artifacts, or
pass the explicit escape hatch:

```bash
--allow-overwrite-artifacts
```

The summary records `preexisting_output_artifacts` either way, and records
`allow_overwrite_artifacts` so later readers can tell whether overwriting was
intentional.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit tests passed `29/29`
- broader focused WorldFoam gate passed `99/99`

Added tests prove:

- a pre-existing output artifact fails before preflight
- the summary lists the stale path and config failure
- the explicit overwrite escape hatch lets a dry-run proceed and records the
  stale path as intentional context

This does not change speed status: the sorted-native emitted-packed path is
still correctness/gate ready, but it still lacks a fresh clean speed promotion.

## Follow-up: post-guard functional MPS smoke

The host remained unsuitable for timing. A fresh preflight still returned
`status=contended`, with high-CPU external blockers including `python -m pytest
tests/`, BTC15M replay/parity jobs, and `shared_feed_daemon.py`.

I ran one tiny functional smoke anyway, explicitly as shader-path evidence and
not speed evidence:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 12 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_postguards_render16_2.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_postguards_render16_2.partial.json
```

Result artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_runtime_smoke_postguards_render16_2.json
```

Result:

- top-level `status=ok`
- `benchmark_environment.status=contended`
- resolved mode `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- `experimental_native_sorted_delta=true`
- `experimental_native_emitted_pack_records_effective=true`
- gradients nonzero
- parameters updated
- outputs finite
- loss decreased
- tiny train/heldout PSNR `12.3217` / `14.5856`
- selected storage split still reports topology `25,272` bytes, coefficient
  sidecar `270,336` bytes, selected total `295,608` bytes

Conclusion: the post-guard path still reaches the real MPS packed framegroup
fused-MSE shader and remains trainable on the tiny fixture. The timings in this
artifact are invalid for speed/sublinear claims because the benchmark
environment was contended.

## Follow-up: stop-and-reflect validation

I stopped variant chasing and ran the current gates without launching another
timing promotion. The promotion wrapper now defaults to requiring two
consecutive clean/background preflight successes before a benchmark launch;
single clean samples are not enough by default.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit tests passed `30/30`
- broader focused WorldFoam gate passed `100/100`

The latest stable promotion attempt still failed before timing with
`preflight_failure_reason=benchmark_environment_never_clean`,
`preflight_required_success_streak=2`, and `preflight_max_success_streak=0`.
So the honest status is: correctness and harness gating are much stronger, the
tiny real shader trainability smoke passed, but there is still no fresh clean
speed/sublinear promotion for sorted-native emitted-packed WorldFoam.

## Follow-up: stable-preflight launch and retryable contamination

I let the real sorted-native emitted-packed promotion wrapper wait for a clean
window instead of launching another known-contaminated run:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_emitted_packed_stable2_wait10m \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 600 \
  --wait-interval-s 15 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records
```

The wrapper first passed the native packed extension verifier. Preflight then
saw transient high-CPU `ai_trader_gpt55_alpha_run2` jobs, recovered, got the
required `2/2` clean/background streak, and launched train/eval.

Artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_packed_stable2_wait10m.json
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_packed_stable2_wait10m.reference_verify.json
research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_packed_stable2_wait10m.promotion_summary.json
```

Outcome:

- train/eval returned top-level `status=ok`
- rows `2,4,8,16` all trained with nonzero gradients, finite outputs,
  parameter updates, and loss decrease
- all rows resolved to
  `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
- record counts and selected storage matched the prior reference exactly
- the verifier failed because the benchmark environment became contended by a
  high-CPU BTC15M verifier at the end of the run
- the failed verifier also reported slow `8f` and `16f` medians versus the
  reference, but that evidence is contaminated and cannot promote or reject the
  shader cleanly

Important numbers from the failed verifier:

- `2f` total/backward median: `1.873 ms` / `1.594 ms`
- `4f` total/backward median: `2.209 ms` / `1.882 ms`
- `8f` total/backward median: `4.723 ms` / `4.257 ms`
- `16f` total/backward median: `7.383 ms` / `6.861 ms`
- contamination: `benchmark_environment status is 'contended'`
- clean promotion: no

I then changed `run_framegroup16_promotion_gate.py` so future runs can retry
contaminated verifier failures without overwriting evidence:

- new `--max-promotion-attempts N`
- retries only when the verifier brief has contamination
- clean verifier regressions still fail immediately
- retries write attempt-suffixed artifacts such as
  `promotion.attempt1.json`, `promotion.attempt1.partial.json`, and
  `promotion.attempt1.reference_verify.json`
- stale artifact protection checks every attempt artifact before launch

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- promotion-wrapper unit tests passed `32/32`
- broader focused WorldFoam gate passed `102/102`

Current honest status: the sorted-native emitted-packed path is correctness
ready and trainable, but the latest real timing sweep is contaminated. The new
wrapper can now keep trying with clean attempt artifacts instead of requiring
manual one-off reruns.

## Follow-up: idle Metal compiler service classifier

The live preflight remained blocked by high-CPU `ai_trader_gpt55_alpha_run2`
activation-bank and pytest jobs, so I did not launch another timing sweep.
While reviewing the failed promotion artifact, I found a smaller gate problem:
the end-of-run environment snapshot could classify an idle system
`MTLCompilerService` process as blocking just because its path contains
`Metal.framework`, even when `pcpu=0.0`.

That is too strict for this lane. A high-CPU Metal compiler service should
still block timing, but an idle system service visible after our own MPS run
should not by itself poison an artifact. I added `mtlcompilerservice` to the
low-CPU background wrapper/service allowlist in
`train_eval_owner_run_tape.py` and added a regression test that:

- accepts idle `MTLCompilerService` at `0.0%` CPU
- still rejects high-CPU `MTLCompilerService` at `95%` CPU
- leaves low-CPU `torch` / `mps` training processes blocking

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_train_eval_benchmark_environment.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed
- focused classifier/wrapper unit tests passed `42/42`
- broader focused WorldFoam gate passed `103/103`

This does not rescue the failed `stable2_wait10m` promotion because that run
also had a high-CPU BTC15M verifier at end-of-run. It does remove a spurious
Metal-service contamination source for the next clean retry.

## Stop-and-reflect checkpoint: 2026-05-19 05:23 +07

We paused shader iteration and re-ran the focused WorldFoam gate after the
latest promotion-wrapper flag pass-through. The gate passed `104/104`:

```bash
rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Current interpretation:

- The sorted-native emitted-packed path is correctness/trainability ready, not
  speed-promoted.
- The clean reference artifact still shows the desired shape:
  total median scale `1.314`, backward median scale `1.364`, storage scale
  `1.040` from `2 -> 16` frames.
- The attempted sorted-native emitted-packed promotion failed verification:
  environment was `contended`, topology storage scale was `2.501`, total median
  scale was `3.942`, backward median scale was `4.304`, and the 8f/16f rows were
  roughly 2x+ slower than the reference.
- The minimal packed-device diagnostic path passed a tiny MPS functional smoke
  with finite outputs, nonzero gradients, parameter updates, loss decrease, and
  PSNR train/heldout `12.322` / `14.586`; its timing was invalid because the
  benchmark environment was still contended.
- The promotion wrapper now forwards
  `--experimental-minimal-packed-delta-device` and related packed-device
  diagnostics, supports retry-on-contamination, writes attempt artifacts, and
  rejects stale output artifacts.

The important lesson is that the math/harness port from STAR-style time tubing
has made WorldFoam frame-group correctness credible, but not yet competitive.
In practice the remaining tax is not just raster math. It is owner-run/order
bookkeeping, topology storage growth, packed-record residency, MPS/Torch
boundary cost, and environment-sensitive timing. Further tweaks should start by
removing one of those taxes cleanly, not by launching another contaminated
promotion loop.

## Launch-only packed native op checkpoint: 2026-05-19 05:38 +0700

After the stop-and-reflect checkpoint, we implemented the lowest-risk native
hot-path split: a launch-only packed framegroup16 fused MSE/VJP op for the
sorted-native emitted-packed delta tape.

What changed:

- Added
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_launch_only_mse_vjp_direct_atomic_rgb_only`
  in the `world_foam_lane2_fused_slab_v0` native extension.
- The new op uses the same Metal kernel body as the existing packed
  framegroup16 direct-atomic path, but takes prepared scalar launch counts from
  the train harness and skips the per-launch `config_i32.cpu()` readback and
  CPU offset/chunk validation scans.
- Wired `--experimental-launch-only-packed-delta` through
  `train_eval_owner_run_tape.py` and `run_framegroup16_promotion_gate.py`.
- Extended the native verifier so promotion preflight requires the launch-only
  op to exist before a launch-only variant can run.

Validation:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 &&
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Native extension rebuild passed.
- Python compile passed.
- Native verifier passed and reports
  `has_launch_only_packed_framegroup16_op: true`.
- Focused promotion/native verifier unit tests passed `34/34`.
- Broader focused WorldFoam gate passed `104/104`.
- A tiny real MPS train/eval smoke passed with:
  - output artifact:
    `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_launch_only_kernel_order_smoke_render16_2.json`
  - `status: ok`
  - resolved tape mode:
    `endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse`
  - `experimental_launch_only_packed_delta: true`
  - `experimental_native_emitted_pack_records_effective: true`
  - nonzero gradients, parameter update, finite outputs, loss decrease
  - train/heldout PSNR `12.322` / `14.586` on the tiny smoke

The speed result is still not promotable. The smoke started clean but ended
contended, and a follow-up benchmark preflight was blocked by two high-CPU
BTC15M row-alignment verifiers plus a high-CPU pytest run. Treat the launch-only
op as a functional/correctness keeper, not as a speed-scale result yet.

Current interpretation:

- This was the right next tax to remove because the existing native path was
  still doing host-side config readback and CPU monotonicity scans on every
  launch.
- It does not address the larger topology/storage shape by itself.
- The next clean comparison should rerun the `2,4,8,16` promotion only after
  the benchmark environment is stable. If that still fails, the bottleneck is
  likely kernel/topology work rather than Python/native launch overhead.

## Resident-storage accounting checkpoint: 2026-05-19 05:45 +0700

We added a first-class split between schema-style selected tape storage and
actual MPS-resident selected-device tensor storage.

What changed:

- `train_eval_owner_run_tape.py` now emits:
  - `train_selected_tape_schema_storage_bytes`
  - `train_selected_tape_schema_topology_storage_bytes`
  - `train_selected_tape_mps_resident_storage_bytes`
  - `train_selected_tape_mps_resident_noncoeff_storage_bytes`
  - `train_endpoint_record_coeff_mps_resident_storage_bytes`
  - top-level first-to-last scales for MPS-resident total, non-coeff, and coeff
    storage.
- `verify_framegroup16_timing_robust.py` now parses/reports those resident
  fields and can reject resident-storage regressions separately from the older
  schema/topology storage checks.
- Tests now cover both the resident tensor-byte split and verifier acceptance /
  rejection of resident-storage scale.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_framegroup16_timing_robust.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Python compile passed.
- Focused storage/verifier tests passed `49/49`.
- Broader focused WorldFoam gate passed `107/107`.
- A tiny launch-only MPS train/eval smoke passed:
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_packed_launch_only_resident_storage_smoke_render16_2.json`
  - status `ok`
  - schema storage `295608`
  - MPS-resident storage `295660`
  - MPS-resident coeff storage `270336`
  - MPS-resident non-coeff storage `25324`
  - train/heldout PSNR `12.322` / `14.586`

The benchmark environment is still contended, now by high-CPU BTC15M verifier /
shadow jobs and a STAR UVT feature overfit run. Do not use the smoke timings as
speed evidence. This checkpoint makes the next clean promotion run more
diagnostic: if total speed still fails, we can distinguish coefficient
residency, non-coeff resident tensor growth, and schema topology growth instead
of collapsing them into one storage number.

## Launch-only variants checkpoint: 2026-05-19 05:53 +0700

Extended the launch-only packed framegroup16 bypass beyond the default packed
mode to the three fused shader variants:

- `recompute`
- `smallrun16`
- `materialized`

The point of this pass was narrow: remove per-launch CPU config readback/scans
from every packed framegroup16 mode so the next clean timing run can test the
shader/topology cost directly.

Validation:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 &&
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment -v
```

Results:

- Rebuild passed.
- Python compile passed.
- Native verifier passed and now requires all four launch-only ops: default,
  recompute, smallrun16, and materialized.
- Focused native/promotion tests passed `34/34`.
- Broader focused WorldFoam gate passed `107/107`.
- Tiny MPS functional smokes passed for all three new variant paths:
  - `recompute`: status `ok`, benchmark environment `contended`, train/heldout
    PSNR `12.322` / `14.586`, first grad abs sum `0.310214`, resident storage
    `295660`, resident non-coeff storage `25324`.
  - `smallrun16`: status `ok`, benchmark environment `contended`, train/heldout
    PSNR `12.322` / `14.586`, first grad abs sum `0.310214`, resident storage
    `295660`, resident non-coeff storage `25324`.
  - `materialized`: status `ok`, benchmark environment `contended`, train/heldout
    PSNR `12.322` / `14.586`, first grad abs sum `0.310214`, resident storage
    `295660`, resident non-coeff storage `25324`.

Interpretation:

- This is a correctness/coverage win, not a speed promotion.
- The fused shader variants are now wired through the same launch-only native
  path, so a future timing result is less likely to be measuring Python/native
  launch bookkeeping.
- The benchmark environment was still contended, so do not cite the smoke
  timings as evidence that WorldFoam is sublinear or competitive with STAR UVT.
- If a clean `2,4,8,16` promotion still fails after this, the remaining problem
  is probably kernel/topology shape rather than launch-side CPU overhead.

## Promotion retry and diagnostic probe extension: 2026-05-19 06:05 +0700

Tried the next clean promotion wrapper for the combined sorted-native emitted
pack-records + kernel-order selected-device + launch-only path:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_emitted_kernelorder_launchonly_clean_promotion \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 600 \
  --wait-interval-s 30 \
  --stable-preflight-checks 2 \
  --max-promotion-attempts 3 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta
```

Result:

- Native packed extension verifier passed before preflight, including all four
  launch-only op booleans.
- Train/eval did not launch.
- Promotion summary:
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_kernelorder_launchonly_clean_promotion.promotion_summary.json`
  - status `preflight_failed`
  - failure reason `stable_preflight_streak_not_reached`
  - required clean streak `2`
  - max clean streak reached `1`
  - attempts `20`
- The blockers were unrelated high-CPU BTC15M / ai_trader Python jobs. Do not
  interpret this as a WorldFoam timing failure.

While the gate waited, I extended
`probe_delta_framegroup_variant_timing.py` so the synthetic MPS op probe can
optionally include:

- packed recompute
- packed smallrun16
- packed materialized
- launch-only default packed
- launch-only recompute
- launch-only smallrun16
- launch-only materialized

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py

rtk .venv/bin/python research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py --help

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py \
  --frame-counts 2 \
  --track-repeats 1 \
  --warmup 0 \
  --steps 1 \
  --no-prewarm-sweep \
  --include-diagnostic-packed-variants \
  --include-launch-only-variants \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_delta_framegroup_launch_only_diagnostic_probe_2f_functional.json
```

The one-step synthetic functional probe passed with exact agreement against the
i16x3 framegroup reference for every included variant:

- all loss diffs: `0.0`
- all grad max diffs: `0.0`
- output artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_delta_framegroup_launch_only_diagnostic_probe_2f_functional.json`

This artifact is not speed evidence. It is a small functional harness proof
that the diagnostic and launch-only packed shader variants can now be exercised
directly without going through full train/eval.

## Focused probe tests: 2026-05-19 06:08 +0700

Added `test_probe_delta_framegroup_variant_timing.py` to pin the new synthetic
probe surface without requiring MPS timing in unit tests.

The tests cover two drift-prone contracts:

- `run_probe(...)` forwards `include_diagnostic_packed_variants` and
  `include_launch_only_variants` into both prewarm frame cases and measured
  frame cases.
- `_packed_launch_only_op(...)` builds the same config tensor metadata and
  scalar metadata tail expected by the launch-only native packed ops.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py \
  research_experiments/world_foam_lane2/test_probe_delta_framegroup_variant_timing.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_delta_framegroup_variant_timing -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_compare_delta_framegroup_i16x3_packed_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate \
  research_experiments.world_foam_lane2.test_verify_framegroup16_timing_robust \
  research_experiments.world_foam_lane2.test_verify_native_packed_extension \
  research_experiments.world_foam_lane2.test_train_eval_benchmark_environment \
  research_experiments.world_foam_lane2.test_probe_delta_framegroup_variant_timing -v
```

Results:

- Python compile passed.
- New focused probe tests passed `2/2`.
- Broader focused WorldFoam gate passed `109/109`.

The benchmark preflight is still contended by unrelated high-CPU `pytest` /
BTC15M verifier jobs, so the real promotion gate remains blocked.

## Promotion rerun reflection: 2026-05-19 07:10 +0700

Stopped the shader iteration loop and reviewed the two promotion attempts that
were meant to decide whether launch-only/kernel-order made the packed path
competitive with the nativechunk directdelta reference.

Sorted-native plus emitted-pack-records plus kernel-order plus launch-only:

- command run id:
  `2026-05-19_sorted_native_emitted_kernelorder_launchonly_clean_promotion_rerun2`
- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_kernelorder_launchonly_clean_promotion_rerun2.attempt1.json`
- verifier:
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_kernelorder_launchonly_clean_promotion_rerun2.attempt1.reference_verify.json`
- verifier status: `failed`
- contamination: `[]`
- 16f median total/backward: `5.182 ms / 4.682 ms`
- robust reference 16f median total/backward: `2.966 ms / 2.640 ms`
- total/backward median scale: `4.585x / 5.526x`
- topology and MPS resident non-coefficient storage scale: `2.501x / 2.500x`
- conclusion: clean negative. Launch-only reduced some prior overhead, but this
  path still regresses 16f versus the existing nativechunk directdelta reference
  and grows non-coefficient/topology storage too much.

Keeper packed path plus kernel-order plus launch-only:

- command run id:
  `2026-05-19_keeper_kernelorder_launchonly_clean_promotion`
- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_keeper_kernelorder_launchonly_clean_promotion.attempt1.json`
- verifier:
  `research_experiments/world_foam_lane2/results/2026-05-19_keeper_kernelorder_launchonly_clean_promotion.attempt1.reference_verify.json`
- verifier status: `failed`
- contamination: `["benchmark_environment status is 'contended'"]`
- median totals by frame: `2f=1.765 ms`, `4f=1.578 ms`, `8f=3.610 ms`,
  `16f=6.416 ms`
- median backwards by frame: `2f=1.190 ms`, `4f=1.042 ms`, `8f=3.148 ms`,
  `16f=5.781 ms`
- total/backward median scale: `3.635x / 4.856x`
- topology and MPS resident non-coefficient storage scale: `2.501x / 2.500x`
- conclusion: not promotable. Because the run ended contended, it is not clean
  speed evidence, but it still failed the same robust thresholds and was worse
  than the clean sorted-native variant at 16f.

The promotion wrapper began waiting for attempt 2 after attempt 1 failed, but
that retry loop was killed at the user's request so we could stop and reflect.
No active promotion process remained after stopping it.

Current read: the math is still sublinear in the narrow sense that 16f is not
8x the 2f cost, and quality/PSNR stays unchanged. But the fused packed variants
are not competitive with the current nativechunk directdelta reference. The
remaining bottleneck is backward/packing overhead plus topology/non-coefficient
storage growth, not image quality. STAR UVT still looks cleaner because its time
tubing keeps the scaled state compact in the hot path, while WorldFoam's packed
delta path still pays too much per-frame metadata/record handling.

## Promotion gate cleanup and minimal-residency probe: 2026-05-19 06:45 +0700

Tightened the promotion wrapper after the failed launch-only reruns:

- `run_framegroup16_promotion_gate.py` now retries contaminated verifier
  failures only when the failures are timing-only. If the verifier also reports
  structural storage/topology failures such as `topology storage scale ...` or
  `MPS resident non-coefficient storage scale ...`, the gate stops after the
  first attempt. This would have avoided the wasted retry loop from the
  contaminated keeper/kernel-order/launch-only attempt.
- The wrapper now rejects
  `--experimental-minimal-packed-delta-device` combined with
  `--experimental-kernel-order-packed-delta-device` at config time, because
  those select different device layouts and should not silently shadow each
  other.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- Python compile passed.
- Focused promotion-gate tests passed `35/35`.

Tried the next unproved variant:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_sorted_native_emitted_minimal_short_preflight \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 60 \
  --wait-interval-s 15 \
  --stable-preflight-checks 2 \
  --max-promotion-attempts 1 \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --experimental-minimal-packed-delta-device
```

Result:

- native packed extension verifier passed
- promotion status: `preflight_failed`
- failure reason: `benchmark_environment_never_clean`
- max clean preflight streak: `0/2`
- no train/eval artifact was produced
- summary:
  `research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_minimal_short_preflight.promotion_summary.json`

This is a blocked run, not a performance result. The variant remains unproved
until the benchmark environment is clean enough to launch timing.

## Minimal-residency guard and launch-only functional smoke: 2026-05-19 07:32 +0700

After the reflection, added a narrow regression guard for the minimal packed
delta fused-device layout:

- `train_eval_owner_run_tape.py` now names the minimal resident tensor contract
  as `_MINIMAL_DELTA_FUSED_DEVICE_TENSOR_KEYS`.
- `test_compare_endpoint_run_record_edit_train_eval.py` now checks that
  `_move_endpoint_record_delta_replace_minimal_fused_tape_to_mps(...)` keeps
  only `frame_t_f32`, `base_offsets_i32`, `track_change_offsets_i32`,
  `change_frame_i32`, and `change_offsets_i32` on MPS. The test explicitly
  rejects accidental residency for `boundary_f32`, `rays_f32`, unpacked
  owner/left/right tables, and change owner/left/right tables.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval -v

rtk env PYTHONPATH=research_experiments/world_foam_lane2:src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- py_compile passed.
- Compare harness tests passed `36/36`; the new MPS layout guard ran and
  passed on this machine.
- Combined compare plus promotion-gate tests passed `71/71`.

Also ran a tiny functional path smoke for the combined
sorted-native/emitted-pack-records/minimal-residency/launch-only path:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-auto-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 8 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_minimal_launchonly_functional_2f_smoke.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_sorted_native_emitted_minimal_launchonly_functional_2f_smoke.partial.json \
  --experimental-native-sorted-delta \
  --experimental-native-emitted-pack-records \
  --experimental-minimal-packed-delta-device \
  --experimental-launch-only-packed-delta
```

Result:

- artifact status: `ok`
- acceptance: `all_rows_ok=true`
- gradients were nonzero, parameters updated, loss decreased, and outputs were
  finite
- selected MPS resident non-coeff storage at this tiny 2f/16px scale:
  `23,452` bytes
- final heldout PSNR at this tiny fixture: `12.777`
- benchmark environment: `contended`

This smoke proves the combined minimal plus launch-only path is functionally
wired. It is not timing evidence; the artifact recorded unrelated pytest and
Metal compiler contention, and it used a 1-step tiny fixture.

## Unchecked launch-only packed framegroup fork: 2026-05-19 08:15 +0700

Added a narrower native fork for the default packed framegroup16 fused-MSE
shader:

- new Metal/C++ op:
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only`
- new train/eval flag:
  `--experimental-unchecked-launch-only-packed-delta`
- the new flag requires `--experimental-launch-only-packed-delta` and is
  restricted to the default packed framegroup16 shader. It relies on the
  existing CPU prepare-time validation and skips the native wrapper's per-launch
  dtype/shape checks before dispatching the same clear + framegroup Metal
  kernels.
- promotion wrapper now forwards the flag and rejects unchecked launch-only
  without checked launch-only.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  python setup.py build_ext --inplace

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval -v

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- py_compile passed.
- native extension rebuilt successfully.
- compare endpoint/run record edit tests passed `36/36`.
- framegroup16 promotion-gate tests passed `36/36`.

Functional MPS smoke:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:research_experiments/world_foam_lane2 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 8 \
  --optimizer-mode manual-vjp \
  --steps 1 \
  --warmup-steps 0 \
  --experimental-launch-only-packed-delta \
  --experimental-unchecked-launch-only-packed-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_unchecked_launch_only_functional_2f_smoke.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_unchecked_launch_only_functional_2f_smoke.partial.json
```

Result:

- artifact status: `ok`
- acceptance: `all_rows_ok=true`
- gradients were nonzero, parameters updated, loss decreased, and outputs were
  finite
- the row records `experimental_launch_only_packed_delta=true` and
  `experimental_unchecked_launch_only_packed_delta=true`
- final heldout PSNR at this tiny fixture: `12.777`
- benchmark environment: `contended`

This is functional wiring evidence only. A clean promotion/timing gate is still
blocked by benchmark preflight; the current preflight reported high-CPU
BTC15M export/build jobs plus a live STAR UVT train process, so this section
does not promote the unchecked launch-only fork as a speed win.

## Unchecked launch-only parity and probe hook: 2026-05-19 08:35 +0700

Added direct regression coverage for the unchecked launch-only fork:

- `test_compare_endpoint_run_record_edit_train_eval.py` now builds a tiny MPS
  packed-framegroup tape and compares the checked launch-only native op against
  the unchecked launch-only native op.
- The test asserts both fused loss and site-RGBA gradients match.
- `probe_delta_framegroup_variant_timing.py` now includes
  `packed_framegroup32_unchecked_launch_only` when
  `--include-launch-only-variants` is passed, so the next clean synthetic sweep
  can compare checked vs unchecked launch overhead directly.

Validation:

```bash
rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval.CompareEndpointRunRecordEditTrainEvalTests.test_unchecked_launch_only_packed_framegroup_matches_checked_launch_only -v

rtk .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python \
  research_experiments/world_foam_lane2/probe_delta_framegroup_variant_timing.py \
  --frame-counts 2 \
  --warmup 0 \
  --steps 1 \
  --include-launch-only-variants \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_unchecked_launch_only_variant_probe_2f_functional.json

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_compare_endpoint_run_record_edit_train_eval -v

rtk env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:src/train .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_framegroup16_promotion_gate -v
```

Results:

- targeted parity test passed.
- variant probe status `ok`, failures `[]`, and
  `packed_framegroup32_unchecked_launch_only` had zero loss/gradient diff
  versus `i16x3_framegroup32_lossreduce`.
- compare endpoint/run tests passed `37/37`.
- framegroup16 promotion-gate tests passed `36/36`.

The owner-run/site-pair next fork still belongs below the Gate4 endpoint delta
tape emission layer (`gate4_affine_slab_tape.py`), not in another in-kernel
boundary-owner toggle. The existing README evidence says in-kernel boundary
pair toggling already lost from register/local-state pressure; the larger
useful fork is to precompute a compact owner-run/site-pair representation while
emitting `Gate4EndpointDeltaReplaceTape`, then add a warm fused-MSE shader that
consumes that precomputed representation.

## Row descriptor launch-only fork: 2026-05-19 07:05 +0700

Added a precomputed row-descriptor packed framegroup16 fork. The CPU helper
`build_delta_replace_frame_row_descriptors(...)` emits one descriptor per real
`(track, frame)`: `row_begin_i32` plus `row_len_source_i16`.
`0x4000` marks change rows and the low bits store row length. The new Metal
kernel
`wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_mse_vjp_direct_atomic_rgb_only_tensor`
uses those descriptors directly, so warm launch no longer scans change-frame
tables inside local-frame 0 before replaying packed records.

Gates run:

- rebuilt `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0`
  with `uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace`
- `py_compile` passed for the touched Python probe/test files
- focused rowdesc + unchecked launch-only tests: `2 passed`
- full `test_compare_endpoint_run_record_edit_train_eval.py`: `38 passed`
- `test_framegroup16_promotion_gate.py`: `36 passed`
- synthetic rowdesc probe:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_launch_only_compact_variant_probe_2_4_8_16_contended.json`

The synthetic probe has `failures=[]` and rowdesc loss/grad deltas are zero
against `i16x3_framegroup32_lossreduce` for 2/4/8/16 frames. Timing is not
promoted: the benchmark preflight was `status=contended` with unrelated high-CPU
BTC15M/test jobs. The compact descriptor row fixes the first padded-storage
mistake: synthetic rowdesc storage is below i16x3 at 2/4/8 frames (`72/128/176`
bytes vs `124/188/188`) and above it at 16 frames (`272` vs `188`). It is still
not a storage promotion against the packed chunk-offset path at 16 frames, but
the shader fork is correct and isolates row-selection work for a future clean
benchmark or descriptor-compression follow-up.

## Rowdesc train/eval gate wiring: 2026-05-19 07:34 +0700

Promoted the rowdesc fork from probe-only into the train/eval harness:

- `train_eval_owner_run_tape.py` now exposes
  `--experimental-rowdesc-launch-only-packed-delta`. It requires
  `--experimental-launch-only-packed-delta`, conflicts with unchecked
  launch-only, and is limited to the default packed framegroup16 fused-MSE
  shader.
- The prepare path builds `row_begin_i32` and `row_len_source_i16` once from the
  delta tape, moves them to MPS, skips `track_chunk_change_offsets_i16` for the
  rowdesc fork, and dispatches
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_launch_only_mse_vjp_direct_atomic_rgb_only`.
- `run_framegroup16_promotion_gate.py` forwards the rowdesc flag and fails fast
  on the same guard conditions.
- `verify_native_packed_extension.py` now checks that the rowdesc launch-only op
  is registered.
- Train/eval rows now save `train_selected_tape_mps_resident_storage_by_key` so
  future gates can see actual MPS residency, not just schema storage.

Validation:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
```

Results:

- `py_compile` passed.
- promotion-gate tests: `38 passed`.
- compare/rowdesc parity tests: `38 passed`.
- native packed extension verifier: `status=ok`,
  `has_launch_only_packed_framegroup16_rowdesc_op=true`.
- functional rowdesc train/eval smoke:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_train_eval_smoke_2f_render16_site4_bykey.json`
  passed with train PSNR `10.884`, heldout PSNR `11.038`, nonzero gradients,
  and parameter updates. It confirms `row_begin_i32` and `row_len_source_i16`
  residency and no `track_chunk_change_offsets_i16`.
- functional rowdesc + kernel-order smoke:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_kernelorder_train_eval_smoke_2f_render16_site4.json`
  passed with the same train/heldout PSNR, nonzero gradients, and parameter
  updates. MPS resident storage was `49,220` bytes, noncoeff `24,644` bytes,
  and resident keys were the packed records, row descriptors, config, frame
  times, and legacy offset tables.

The benchmark environment was still `status=contended` for the functional
smokes, so timing remains path-smoke-only. The next real promotion gate should
run the rowdesc + kernel-order flags on an idle machine with the full 2/4/8/16
frame ladder and reference verifier.

## Rowdesc kernel-order hot residency strip: 2026-05-19 07:52 +0700

The first rowdesc + kernel-order smoke still carried legacy row-selection
tables on MPS (`base_offsets_i32`, `track_change_offsets_i32`,
`change_frame_i32`, `change_offsets_i32`) even though the rowdesc shader does
not read them. Tightened the train/eval harness:

- `_delta_replace_coeff16_fused_mse_loss_vjp(...)` now allows the rowdesc
  launch-only path to execute without `change_offsets_i32`; non-rowdesc delta
  paths still fail loudly without the legacy tables.
- The kernel-order prepare path skips the four legacy offset tensors when
  `--experimental-rowdesc-launch-only-packed-delta` is active.
- `_render_device_for_tape(...)` now lazily materializes the full replay device
  for minimal rowdesc train/eval final rendering, so PSNR rendering remains
  covered even though the warm VJP device is stripped.

Validation:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 4 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-rowdesc-launch-only-packed-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_kernelorder_minresident_train_eval_smoke_2f_render16_site4.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_kernelorder_minresident_train_eval_smoke_2f_render16_site4.partial.json
```

Results:

- `py_compile` passed.
- compare/rowdesc parity tests: `38 passed`.
- promotion-gate tests: `38 passed`.
- min-resident rowdesc + kernel-order smoke: `status=ok`, train PSNR `10.884`,
  heldout PSNR `11.038`, nonzero gradients, and parameter updates.
- Hot MPS resident storage dropped from the prior rowdesc + kernel-order smoke
  `49,220` bytes (`24,644` noncoeff) to `41,016` bytes (`16,440` noncoeff).
  Resident keys are now only `delta_base_record_i32`,
  `delta_change_record_i32`, `delta_coeff_f16`, `delta_config_f32`,
  `delta_config_i32`, `frame_t_f32`, `row_begin_i32`, and
  `row_len_source_i16`.

Benchmark environment remains `status=contended`, so this is correctness and
residency evidence, not a speed promotion. The full clean gate should use the
same rowdesc + kernel-order flags once the machine is idle.

## Rowdesc schema storage accounting fix: 2026-05-19 08:08 +0700

After the hot residency strip, the rowdesc smoke still reported
`train_selected_tape_schema_storage_bytes=49,168` because
`_selected_tape_storage_bytes(...)` was counting legacy delta index tables even
when the rowdesc representation no longer included them. Fixed the accounting
so rowdesc selected schema storage counts packed records plus row descriptors
and coeff storage, not `base_offsets_i32`, `track_change_offsets_i32`,
`change_frame_i32`, or `change_offsets_i32`.

Validation:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 4 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-rowdesc-launch-only-packed-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_kernelorder_minresident_schema_smoke_2f_render16_site4.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_kernelorder_minresident_schema_smoke_2f_render16_site4.partial.json
```

Results:

- `py_compile` passed.
- compare/rowdesc parity tests: `39 passed`.
- promotion-gate tests: `38 passed`.
- schema-correct min-resident rowdesc smoke: `status=ok`, train PSNR `10.884`,
  heldout PSNR `11.038`, nonzero gradients, and parameter updates.
- Schema storage is now `40,964` bytes and schema topology storage is `16,388`
  bytes, closely matching MPS resident storage `41,016` bytes and noncoeff
  resident storage `16,440` bytes. The 52-byte difference is `frame_t_f32` plus
  launch config tensors, which are resident runtime inputs rather than schema
  record payload.

Benchmark environment remains `status=contended`; do not use this smoke for
speed claims.

## Rowdesc unchecked launch-only fork: 2026-05-19 09:02 +0700

Added a narrow prepare-validated rowdesc unchecked launch-only native op:
`endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only`.
It uses the same rowdesc Metal kernel and launch descriptors as the checked
rowdesc path, but skips the per-call C++ dtype/shape/count validation under the
same contract as the older non-rowdesc unchecked launch-only op. The train path
now selects it when both `--experimental-rowdesc-launch-only-packed-delta` and
`--experimental-unchecked-launch-only-packed-delta` are active; the promotion
wrapper no longer rejects that flag combination.

Validation:

```bash
rtk zsh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 4 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-unchecked-launch-only-packed-delta \
  --experimental-rowdesc-launch-only-packed-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_smoke_2f_render16_site4.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_smoke_2f_render16_site4.partial.json

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_smoke_2f_render16_site4.json \
  --expected-frames 2 \
  --expected-tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_smoke_2f_render16_site4.verify.json
```

Results:

- native extension rebuild passed.
- native extension verifier reports
  `has_launch_only_packed_framegroup16_rowdesc_unchecked_op=true`.
- rowdesc parity node passed checked, unchecked, rowdesc, and
  rowdesc-unchecked agreement (`1 passed`), then the full compare gate passed
  `39 passed`.
- promotion-gate tests passed `38 passed`, including dry-run forwarding of
  rowdesc plus unchecked flags.
- rowdesc-unchecked train/eval smoke: `status=ok`, train PSNR `10.884`,
  heldout PSNR `11.038`, nonzero gradients, and parameter updates. It keeps
  schema storage `40,964` bytes, MPS resident storage `41,016` bytes, and
  noncoeff resident storage `16,440` bytes with the same compact resident keys
  as the checked rowdesc path.
- robust verifier on the smoke correctly surfaces resident storage scales as
  `1.0`, but returns `status=failed` only because
  `benchmark_environment.status=contended`.

This is a correctness/path-readiness fork, not a speed promotion. The next
clean speed gate should use rowdesc + unchecked + kernel-order on a clean
machine, then compare against the current packed framegroup16 direct-delta
reference.

## Rowdesc unchecked verifier identity contract: 2026-05-19 07:38 +0700

The benchmark preflight is still contended by unrelated high-CPU pytest/BTC
jobs, so no clean 2/4/8/16 promotion was launched. Instead, tightened the
promotion verifier so the next clean artifact must prove it actually used the
intended shader fork, not merely the same tape mode.

`verify_framegroup16_timing_robust.py` now accepts repeated
`--expect-payload-bool KEY=BOOL` checks. Each expected boolean must match at
top level and on every row. `run_framegroup16_promotion_gate.py` forwards all
enabled train variant flags into the verifier command, so a rowdesc-unchecked
promotion now verifies:

- `experimental_kernel_order_packed_delta_device=true`
- `experimental_launch_only_packed_delta=true`
- `experimental_unchecked_launch_only_packed_delta=true`
- `experimental_rowdesc_launch_only_packed_delta=true`

Validation:

```bash
rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/test_verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py

rtk env PYTHONPATH=research_experiments/world_foam_lane2 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_verify_framegroup16_timing_robust.py -q

rtk env PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_smoke_2f_render16_site4.json \
  --expected-frames 2 \
  --expected-tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --expect-payload-bool experimental_kernel_order_packed_delta_device=true \
  --expect-payload-bool experimental_launch_only_packed_delta=true \
  --expect-payload-bool experimental_unchecked_launch_only_packed_delta=true \
  --expect-payload-bool experimental_rowdesc_launch_only_packed_delta=true \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_smoke_2f_render16_site4.expected_flags_verify.json

rtk env PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id unit_rowdesc_unchecked_expected_flags_dryrun \
  --summary-json /tmp/rowdesc_unchecked_expected_flags_promotion_summary.json \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-unchecked-launch-only-packed-delta \
  --experimental-rowdesc-launch-only-packed-delta \
  --no-reference-artifact \
  --dry-run
```

Results:

- verifier unit tests passed `16 passed`.
- promotion-gate tests passed `38 passed`.
- expected-flag verifier on the existing rowdesc-unchecked smoke reports all
  four expected booleans in `expected_payload_bools`; it still returns
  `status=failed` only because the artifact's benchmark environment is
  contended.
- dry-run promotion summary status is `ok` and its `verify_command` includes
  the four expected boolean checks listed above.

The next clean run should use the same promotion wrapper command without
`--dry-run` once preflight returns clean/background.

## Rowdesc unchecked refresh under contended machine: 2026-05-19 11:22 +0700

Stopped short of another full 2/4/8/16 promotion because the benchmark
preflight is still dirty. Current blocker was a high-CPU `pytest tests/`
process; the fresh smoke also saw multiple `ai_trader` verification jobs plus a
STAR UVT feature train consuming CPU, so no timing number from this section is
promotion evidence.

Current-tree verification that did run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/verify_framegroup16_timing_robust.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_verify_framegroup16_timing_robust.py -q
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py::CompareEndpointRunRecordEditTrainEvalTests::test_unchecked_launch_only_packed_framegroup_matches_checked_launch_only -q
```

Results: `py_compile` passed, verifier tests `16 passed`, promotion-wrapper
tests `38 passed`, native packed extension verify reported `status=ok` with
`has_launch_only_packed_framegroup16_rowdesc_unchecked_op=true`, and the focused
checked/unchecked/rowdesc parity node passed.

Also ran a fresh 2f/16px/4-site current-tree rowdesc-unchecked smoke:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 4 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-unchecked-launch-only-packed-delta \
  --experimental-rowdesc-launch-only-packed-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_current_smoke_2f_render16_site4.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc_unchecked_kernelorder_minresident_current_smoke_2f_render16_site4.partial.json
```

The smoke is functional: `status=ok`, train PSNR `10.884151190417564`,
heldout PSNR `11.038369688720046`, gradients nonzero, parameters updated, and
the row reports all four rowdesc-unchecked/kernel-order flags. Storage fields
are still the useful split: schema `40964`, topology `16388`, MPS resident
`41016`, MPS resident non-coeff `16440`, coefficient MPS resident `24576`.
The verifier with `--expect-payload-bool` found no structural failures and all
expected flags true, but returned failed solely because
`benchmark_environment.status=contended`.

The important state: current build and identity gates are healthy; speed
promotion is still pending on an idle machine. The next clean command remains
the rowdesc + unchecked + kernel-order promotion wrapper with stable preflight.

## Rowdesc32 shader fork smoke: 2026-05-19 11:48 +0700

Added a separate rowdesc32 launch-only packed framegroup16 shader fork instead
of silently changing the existing rowdesc kernel. The fork keeps the same
rowdesc ABI (`row_begin_i32`, `row_len_source_i16`, packed base/change records)
but increases the threadgroup site-gradient reduction slots from 16 to 32. The
target use case is the 24-site promotion shape, where the old rowdesc path
falls back to per-segment global atomics.

Changed surfaces:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_mse_vjp_direct_atomic_rgb_only_tensor`
- Torch ops:
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_launch_only_mse_vjp_direct_atomic_rgb_only`
  and
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_rowdesc32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only`
- train/eval flag:
  `--experimental-rowdesc32-launch-only-packed-delta`
- native verifier booleans:
  `has_launch_only_packed_framegroup16_rowdesc32_op=true` and
  `has_launch_only_packed_framegroup16_rowdesc32_unchecked_op=true`

Verification run:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py::CompareEndpointRunRecordEditTrainEvalTests::test_unchecked_launch_only_packed_framegroup_matches_checked_launch_only -q
```

Results: native build passed; `py_compile` passed; native packed extension
verify reported `status=ok` with the new rowdesc32 booleans; verifier +
promotion-gate tests passed `40 passed`; focused MPS launch-only parity passed
`1 passed`. The parity fixture now uses `site_count=24`, so rowdesc32 exercises
the 32-slot reduction case that the 16-slot rowdesc shader cannot reduce.

Functional smoke:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 24 \
  --steps 1 \
  --warmup-steps 0 \
  --optimizer-mode manual-vjp \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --endpoint-record-source gate4-affine \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-unchecked-launch-only-packed-delta \
  --experimental-rowdesc-launch-only-packed-delta \
  --experimental-rowdesc32-launch-only-packed-delta \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc32_unchecked_kernelorder_minresident_smoke_2f_render16_site24.json \
  --partial-out-json research_experiments/world_foam_lane2/results/2026-05-19_rowdesc32_unchecked_kernelorder_minresident_smoke_2f_render16_site24.partial.json
```

Smoke result: `status=ok`, train PSNR `12.589492913772977`, heldout PSNR
`14.948773035163148`, gradients nonzero, parameters updated, and both payload
and row flags report rowdesc32 + rowdesc + unchecked + kernel-order true.
Selected train MPS resident storage split at 2f/16px/24-site was total
`1156852`, non-coeff `26356`, coeff `1130496`; rowdesc descriptor residency was
`row_begin_i32=4096` and `row_len_source_i16=2048`.

The robust timing verifier with all expected payload booleans found
`failures=[]`, but returned `status=failed` solely because
`benchmark_environment.status=contended`. Contending processes included
high-CPU `pytest`/`ai_trader` jobs, so the smoke timing (`total_ms=107.565`,
`backward_ms=6.409`) is not promotion evidence. The next required step is a
clean 2/4/8/16 promotion run with this rowdesc32 flag enabled once preflight is
idle.

## 2026-05-19 Reduce32 pause and reflection

The clean rowdesc32 promotion was the important negative result. It proved the
rowdesc32 fork is functional and uncontaminated, but it did not pass the gate:
`topology_storage_scale=3.908031293417752`,
`mps_resident_noncoeff_storage_scale=3.907809353322891`, and the verifier
reported total/backward scale failures plus 8f/16f reference regressions. Total
storage scale stayed near flat at `1.0673689149397945`, so the coefficient tape
packing is doing the expected work. The failure is specifically the per-frame
topology/non-coefficient sidecar introduced by row descriptors.

The rowdesc16 comparison did not produce a clean measurement. Its promotion
wrapper timed out in preflight with
`preflight_failure_reason=stable_preflight_streak_not_reached`,
`preflight_max_success_streak=1`, so there is no fair rowdesc16 speed row to
promote.

Based on that failure mode, I added a compact `reduce32` fork instead of
continuing to push rowdesc. It keeps the existing compact chunk-offset ABI, but
uses a 32-slot site-gradient threadgroup reduction for the 24-site promotion
shape. The new surfaces are:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_mse_vjp_direct_atomic_rgb_only_tensor`
- Torch ops:
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_launch_only_mse_vjp_direct_atomic_rgb_only`
  and
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_reduce32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only`
- train/eval flag:
  `--experimental-reduce32-launch-only-packed-delta`

Verification for the fork passed: native build passed, `py_compile` passed,
native extension verification reported the reduce32 ops, and the focused pytest
set passed `44 passed`. A 2f functional smoke also passed with nonzero
gradients and parameter updates:

- train PSNR: `12.589492913772977`
- heldout PSNR: `14.948773035163148`
- selected MPS resident storage: `1160960`
- selected MPS resident non-coeff storage: `30464`

That smoke was timing-contended, so its `total_ms=182.88433400448412` and
`backward_ms=8.03008396178484` are not performance evidence. The verifier
reported `failures=[]`; the failed status was due to benchmark environment
contamination only.

Current interpretation: rowdesc32 is a clean negative result for promotion
because it restores correctness at 24 sites but pays too much per-frame
topology residency. Reduce32 is the more coherent next candidate because it
preserves the compact ABI and only changes the reduction width. The remaining
open gate is a clean reduce32 2/4/8/16 promotion run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_reduce32_unchecked_kernelorder_minresident_promotion_2_4_8_16 \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-unchecked-launch-only-packed-delta \
  --experimental-reduce32-launch-only-packed-delta \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 600 \
  --wait-interval-s 30 \
  --stable-preflight-checks 2 \
  --max-promotion-attempts 2
```

## 2026-05-19 Rowselect32 zero-storage rowdesc fork

Tried to run the clean reduce32 promotion first, but the wrapper failed before
train/eval because the machine never reached two consecutive clean preflights:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_reduce32_unchecked_kernelorder_minresident_promotion_2_4_8_16.promotion_summary.json`
- status: `preflight_failed`
- reason: `stable_preflight_streak_not_reached`
- required/current/max streak: `2/0/1`
- top blockers were unrelated `ai_trader` pytest/RL/export jobs.

While waiting on the machine, the code-level conclusion was that reduce32 only
changes site-gradient aggregation width. If clean reduce32 still fails speed, the
next useful test is not another rowdesc storage format, but a zero-storage
version of rowdesc's useful idea: compute each frame lane's `begin/end/source`
inside the hot kernel from the existing compact chunk-offset tables, instead of
having `local_frame == 0` serially preselect all row bounds or materializing
per-track-frame row descriptors.

Implemented that as `rowselect32`:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_mse_vjp_direct_atomic_rgb_only_tensor`
- Torch ops:
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_launch_only_mse_vjp_direct_atomic_rgb_only`
  and
  `endpoint_record_delta_replace_coeff16_packed_framegroup16_rowselect32_unchecked_launch_only_mse_vjp_direct_atomic_rgb_only`
- train/eval flag:
  `--experimental-rowselect32-launch-only-packed-delta`

The first rowselect smoke accidentally showed why row-level payload flags matter:
the top-level payload flag was true, but rows still had rowselect false because
`run_train_eval()` was not forwarding the new option into `_run_one()`. Fixed
that and reran as `smoke2`; both top-level and row-level rowselect flags are now
true.

Verification:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && \
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_framegroup16_promotion_gate.py -q
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py::CompareEndpointRunRecordEditTrainEvalTests::test_unchecked_launch_only_packed_framegroup_matches_checked_launch_only -q
```

Results: native build passed; `py_compile` passed; native verifier reported
`has_launch_only_packed_framegroup16_rowselect32_op=true` and
`has_launch_only_packed_framegroup16_rowselect32_unchecked_op=true`;
promotion-gate/native tests passed `47 passed`; focused MPS parity passed
`1 passed`; the rowselect32 dry-run routing test passed after the forwarding
fix.

Functional smoke:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowselect32_unchecked_kernelorder_minresident_smoke2_2f_render16_site24.json`
- verifier:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowselect32_unchecked_kernelorder_minresident_smoke2_2f_render16_site24.verify.json`
- status: `ok`
- top-level and row-level `experimental_rowselect32_launch_only_packed_delta`: `true`
- train PSNR: `12.589492913772977`
- heldout PSNR: `14.948773035163148`
- first grad abs sum: `0.33094555139541626`
- selected MPS non-coeff storage: `30464`
- selected topology storage: `30412`

The robust verifier found `failures=[]`, but returned `status=failed` only
because `benchmark_environment.status=contended`. The smoke timing
(`total_ms=46.177`, `backward_ms=6.426`) is not speed evidence.

Next clean promotion command, once preflight is idle:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/run_framegroup16_promotion_gate.py \
  --run-id 2026-05-19_rowselect32_unchecked_kernelorder_minresident_promotion_2_4_8_16 \
  --tape-mode endpoint-record-delta-replace-coeff16-packed-framegroup16-fused-mse \
  --experimental-kernel-order-packed-delta-device \
  --experimental-launch-only-packed-delta \
  --experimental-unchecked-launch-only-packed-delta \
  --experimental-rowselect32-launch-only-packed-delta \
  --wait-for-benchmark-environment-ok \
  --wait-timeout-s 600 \
  --wait-interval-s 30 \
  --stable-preflight-checks 2 \
  --max-promotion-attempts 2
```

## Rowselect32 promotion reflection

The clean-rowselect promotion wrapper did get through train/eval for 2/4/8/16
frames, but it did not promote:

- artifact:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowselect32_unchecked_kernelorder_minresident_promotion_2_4_8_16.attempt1.json`
- verifier:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowselect32_unchecked_kernelorder_minresident_promotion_2_4_8_16.attempt1.reference_verify.json`
- summary:
  `research_experiments/world_foam_lane2/results/2026-05-19_rowselect32_unchecked_kernelorder_minresident_promotion_2_4_8_16.promotion_summary.json`
- verifier status: `failed`
- contamination: `benchmark_environment status is 'contended'`

The correctness/routing result was good: the native ops existed, the rowselect
smoke exercised the real MPS path, and row-level payload flags confirmed the
rowselect32 route after fixing option forwarding into `_run_one()`.

The scaling result was not good enough:

- total storage scale was almost flat: `1.0399x`, because `delta_coeff_f16`
  dominates total bytes.
- topology storage scale was `2.5005x`, failing the `<=1.1x` gate.
- MPS resident non-coeff storage scale was `2.5005x`, failing the `<=1.1x`
  gate.
- timing was not clean benchmark evidence because the final environment was
  contended, but the verifier still reported `backward_median_scale=2.0777x`
  and reference speed failures at 2f, 8f, and 16f.

Per-row saved values:

- 2f: total median `3.385 ms`, backward median `2.727 ms`, heldout PSNR
  `14.1702`, topology `494,520 B`, noncoeff `494,572 B`.
- 4f: total median `2.151 ms`, backward median `1.740 ms`, heldout PSNR
  `13.9925`, topology `763,028 B`, noncoeff `763,088 B`.
- 8f: total median `6.562 ms`, backward median `6.008 ms`, heldout PSNR
  `14.2205`, topology `1,084,612 B`, noncoeff `1,084,688 B`.
- 16f: total median `5.998 ms`, backward median `5.665 ms`, heldout PSNR
  `14.2321`, topology `1,236,560 B`, noncoeff `1,236,668 B`.

Interpretation: rowselect32 removed the row-description sidecar as the obvious
culprit, but the packed delta-record representation still carries
frame-scaling topology through `delta_change_record_i32`, `change_frame_i32`,
and `change_offsets_i32`. This makes it a useful fork but not the STAR-like
answer. The next serious fork should pivot away from more delta-record variants
and toward the Gate4 affine/candidate CSR path (`row_offsets`, `candidate_ids`,
`candidate_depth_coeffs`) where topology can be binned once and replayed across
time, closer to STAR UVT's tile-time ABI.

## Gate4 affine candidate CSR fused-MSE train/eval bridge

Implemented a first-class train/eval tape mode:

```text
gate4-affine-candidate-num32-den16-fused-mse
```

This path builds `Gate4AffineSlabTape`, keeps only the candidate CSR / affine
ray tensors resident on MPS, trains with the existing native fused kernel
`fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only`, and renders
PSNR with `fused_slab_affine_num32_den16_realray_rgba_depth_replay`.

Files touched:

- `research_experiments/world_foam_lane2/train_eval_owner_run_tape.py`
- `research_experiments/world_foam_lane2/verify_native_packed_extension.py`
- `research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py`
- `research_experiments/world_foam_lane2/test_verify_native_packed_extension.py`

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/verify_native_packed_extension.py \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_compare_endpoint_run_record_edit_train_eval.py::CompareEndpointRunRecordEditTrainEvalTests::test_affine_candidate_resident_storage_is_not_counted_as_coeff_sidecar -q
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_verify_native_packed_extension.py -q
```

Results: `py_compile` passed; focused affine storage unit passed `1 passed`;
native verifier test passed `1 passed` and now requires
`has_affine_candidate_num32_den16_fused_mse_op=true`.

Tiny MPS correctness smoke:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --tape-mode gate4-affine-candidate-num32-den16-fused-mse \
  --optimizer-mode manual-vjp \
  --frame-counts 2 \
  --render-size 16 \
  --site-count 24 \
  --steps 2 \
  --warmup-steps 1 \
  --defer-heldout-device \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_fused_mse_smoke2_2f_render16_site24.json
```

Result: status `ok`, train PSNR `13.7439`, heldout PSNR `15.2406`,
first grad abs sum `0.3309276`, max parameter update `0.089588`, fused
backward mean `3.845 ms`, total mean `4.415 ms`. Timing is only smoke-level;
a STAR UVT process was active around this work, so do not cite this as a clean
speed benchmark.

Storage scale probe without MPS timing:

- 2f: candidate count `84,930`, max candidates/row `224`, storage
  `1,048,324 B`
- 4f: candidate count `84,609`, max candidates/row `222`, storage
  `1,044,480 B`
- 8f: candidate count `84,196`, max candidates/row `221`, storage
  `1,039,540 B`
- 16f: candidate count `84,225`, max candidates/row `220`, storage
  `1,039,920 B`
- storage scale 2f->16f: `0.99198x`
- candidate scale 2f->16f: `0.99170x`

Interpretation: unlike the packed delta-record rowselect32 path, the candidate
CSR representation has the right sublinear topology shape across frame count on
this fixed moving-camera fixture. The remaining unknown is runtime: we still
need a clean 2/4/8/16 MPS benchmark after the active STAR UVT run is out of the
way. If runtime is not competitive, the next likely bottleneck is candidate
replay work per frame, not resident topology growth.

## Candidate CSR 2/4/8/16 speed gate and verifier

Added a focused verifier:

- `research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py`
- `research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py`

The verifier checks:

- benchmark/tape mode is the candidate CSR fused-MSE path
- frame rows match the requested 2/4/8/16 set
- all row acceptance flags are true
- gradients, updates, train/heldout PSNR, and candidate counts are positive
- max candidates/row stays under the 256 fused-MSE cap
- resident storage, resident noncoeff storage, and candidate count scale stay
  under `1.10x`
- total/backward mean and median scale stay under `2.0x`
- `benchmark_environment.status` must be `background` unless
  `--allow-contended` is passed

Verification:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py
PYTHONPATH=research_experiments/world_foam_lane2 PYTHONDONTWRITEBYTECODE=1 \
  uv run --with pytest python -m pytest \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_candidate_csr_train_eval.py -q
```

Result: `py_compile` passed; verifier tests passed `4 passed`.

Clean-start 2/4/8/16 candidate CSR run:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  --require-benchmark-environment-ok \
  --tape-mode gate4-affine-candidate-num32-den16-fused-mse \
  --optimizer-mode manual-vjp \
  --frame-counts 2,4,8,16 \
  --render-size 16 \
  --site-count 24 \
  --steps 5 \
  --warmup-steps 2 \
  --defer-heldout-device \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_fused_mse_scale_2_4_8_16_render16_site24.json
```

Artifact status was `ok`, rows were all `ok`, but the final benchmark
environment became `contended` because an unrelated high-CPU `ai_trader`
process appeared before the end snapshot. Therefore this is not a promoted
clean-speed artifact.

Diagnostic verifier with contamination allowed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_candidate_csr_train_eval.py \
  research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_fused_mse_scale_2_4_8_16_render16_site24.json \
  --allow-contended \
  --out-json research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_fused_mse_scale_2_4_8_16_render16_site24.verify_allow_contended.json
```

Result: `status=ok` with contamination noted. Strict verifier without
`--allow-contended` wrote
`research_experiments/world_foam_lane2/results/2026-05-19_gate4_affine_candidate_csr_fused_mse_scale_2_4_8_16_render16_site24.verify.json`
and failed only on `benchmark_environment status is 'contended'`.

Diagnostic numbers:

- 2f: total mean `4.588 ms`, backward mean `4.084 ms`, heldout PSNR
  `14.1880`, candidates `84,930`, storage `1,048,324 B`
- 4f: total mean `5.116 ms`, backward mean `4.518 ms`, heldout PSNR
  `14.2189`, candidates `84,609`, storage `1,044,480 B`
- 8f: total mean `3.600 ms`, backward mean `3.122 ms`, heldout PSNR
  `14.6554`, candidates `84,196`, storage `1,039,540 B`
- 16f: total mean `4.311 ms`, backward mean `3.836 ms`, heldout PSNR
  `14.8117`, candidates `84,225`, storage `1,039,920 B`

Scales from the diagnostic artifact:

- total mean scale 2f->16f: `0.9397x`
- total median scale 2f->16f: `0.9970x`
- backward mean scale 2f->16f: `0.9392x`
- backward median scale 2f->16f: `1.0234x`
- resident storage scale: `0.9920x`
- resident noncoeff storage scale: `0.9920x`
- candidate count scale: `0.9917x`

Interpretation: this is the first WorldFoam path in this lane with both
STAR-like topology storage and STAR-like apparent runtime scale across
2/4/8/16 moving-camera frames. Because the end snapshot was contended, the
artifact is diagnostic rather than promotable. The next gate is simply a clean
rerun through the strict verifier. If that passes, the next real comparison is
against the matching STAR UVT artifact, not another delta-record microfork.
