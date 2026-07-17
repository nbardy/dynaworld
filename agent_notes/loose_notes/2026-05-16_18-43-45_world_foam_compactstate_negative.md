# WorldFoam Compact-State Framegroup Negative

Context: tested a smaller local-state rewrite of the live
`endpoint-record-delta-replace-coeff16-i16x3-framegroup16-fused-mse` kernel.
The hypothesis was that carrying fewer per-segment arrays inside each frame lane
would improve occupancy/local-memory pressure in the current row-reference /
loss-reduced path.

Edit tried in
`third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`:

- removed the per-run `segment_alpha`, `weights`, and `segment_rgb` arrays from
  the i16x3 framegroup16 fused-MSE kernel
- kept only `owner`, `length`, `trans_before`, and `segment_trans`
- recomputed `alpha`, `weight`, and reloaded RGB during the reverse pass

Correctness:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk -v
```

The focused parity gate passed after cleaning two mechanical edit mistakes.

Timing artifacts:

```text
research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_compactstate_variant_timing_probe_tracks128_interleaved_warm3_steps8_16_32_64_128.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_compactstate_fused_mse_repeat_loaded_warm5_steps12_render32_site12_16_32_64_128.partial.json
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup16_compactstate_fused_mse_short_sweep_warm1_steps3_render32_site12_16_32_64_128.partial.json
```

Isolated synthetic op probe:

- 16/32/64/128 mean ms: `4.590 / 3.932 / 65.422 / 2.463`
- 16/32/64/128 median ms: `3.620 / 4.360 / 3.543 / 2.516`
- the 128f row looked clean, but 64f had a `498.955 ms` max outlier

Train/eval partials:

- warm5/steps12 render32/site12 16f row: total/backward
  `4.384 / 3.205 ms`, heldout PSNR `14.6291`
- warm1/steps3 render32/site12 16f row: total/backward
  `4.477 / 3.001 ms`, heldout PSNR `12.5775`

Interpretation: this is a runtime negative. The compact-state rewrite reduces
local arrays, but recomputing/reloading RGB/alpha/weight in the reverse pass
does not improve the train/eval path and worsens the 16f row versus the current
loss-reduced winner (`3.046 / 2.510 ms` in the saved render32/site12 sweep).
The full sweep was intentionally stopped after the 16f row twice; it was not
worth spending 64/128 train/eval time on a slower 16f gate.

Current state after this note:

- compact-state shader edit was reverted
- focused framegroup parity passed again: 2 tests OK
- full world-foam lane passed again:
  `PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q`
  reported 88 tests OK
- no train/probe processes were left running

Next implication: do not chase register/local-array shrinkage by adding reverse
pass reloads. The current i16x3 row-reference/loss-reduced kernel remains the
practical winner. Better next candidates should change the amount of work
launched or the producer-side topology, not merely reshuffle per-frame local
state.
