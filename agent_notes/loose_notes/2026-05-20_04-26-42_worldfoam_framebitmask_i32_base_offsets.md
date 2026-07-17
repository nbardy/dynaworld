# WorldFoam framebitmask int32 base offsets

Context: after the native owner-run cutwalk and repeated-fixture 32f gate, a
larger render96/site48 diagnostic smoke hit a real remaining shader blocker:
the factorized framebitmask path still projected `base_offsets_i32` to int16.
At this pressure the max base offset was `83695`, so the old path failed before
writing an artifact.

What changed:

- `owner-run-delta-packed-factorized-framebitmask-recompute-fused-mse-nomid`
  now keeps `base_offsets_i32` on the selected device instead of
  `base_offsets_i16`.
- The Python wrapper, torch library schema, C++ MPS dispatch checks, and Metal
  framebitmask fused-MSE shader now treat the base-offset buffer as int32.
- Storage accounting was updated so the selected schema includes
  `base_offsets_i32`, `track_change_offsets_i32`, `change_offsets_i32`, and
  `track_frame_mask_i32` without negative `unattributed_storage`.
- Added a regression that builds a render96/site48/2f native-cutwalk
  framebitmask tape and requires `base_offsets_i32.max() > 32767`, no
  `base_offsets_i16`, and clean schema byte accounting.

Validation:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_factorized_framebitmask_removes_dense_frame_table \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_framebitmask_keeps_i32_base_offsets_when_offsets_exceed_i16 \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_native_cutwalk_framebitmask_shader_output_matches_python_for_multiview_moving_rays -v
```

Result: `Ran 3 tests in 90.283s`, `OK`.

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.NativeOwnerRunCutwalkCpuTests.test_framebitmask_supports_frame31_signed_int32_payload_for_32_frames \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.NativeOwnerRunCutwalkCpuTests.test_framebitmask_still_rejects_more_than_32_frames -v
```

Result: `Ran 2 tests in 0.000s`, `OK`.

```bash
PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_selected_only_framebitmask_prep_skips_baseline_segment_tape \
  research_experiments.world_foam_lane2.test_train_eval_owner_run_delta_packed.OwnerRunDeltaPackedTrainEvalTests.test_native_cutwalk_framebitmask_matches_python_sequence_shader_output -v
```

Result: `Ran 2 tests in 11.795s`, `OK`.

The larger smoke now writes:

```text
research_experiments/world_foam_lane2/results/2026-05-20_worldfoam_native_cutwalk_render96_site48_2f_functionality_smoke.json
```

Key artifact fields: `status=ok`, `row_status=ok`,
`base_offsets_i32.max=83695`, `base_offsets_i32.eligible=false`, no
`unattributed_storage`, and schema keys
`base_offsets_i32/base_record_packed/change_offsets_i32/change_record_packed/factorized_coeff_f32/track_change_offsets_i32/track_frame_mask_i32`.

Do not cite the smoke as promotable timing evidence. Its
`benchmark_environment.status` is `contended` because unrelated ai_trader
monitor/export work was active. The smoke is correctness and accounting
evidence for larger packed-record metadata.

One bad unittest invocation used the wrong class name for the two CPU
framebitmask helper tests and ended with loader errors even though the selected
MPS tests in that invocation passed. The corrected CPU helper command above is
the one to cite.
