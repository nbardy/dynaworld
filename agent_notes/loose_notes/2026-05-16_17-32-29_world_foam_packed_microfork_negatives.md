# World Foam Packed Framegroup Microfork Negatives

Context: continued the packed-record framegroup16 fused-MSE lane after the broad
64/128 interleaved guard rejected packed promotion at 128f. The target was to
see whether the 128f loss came from redundant packed-record decode or sentinel
cut helper overhead inside the packed Metal loop.

Tried two shader edits in
`third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`.
Both were reverted after measurement.

1. Threadgroup predecode for packed records:
   - Added a fast path that detects chunks where all frame lanes share the same
     selected packed row, decodes owner/left/right once into threadgroup arrays,
     and lets all lanes reuse those decoded values.
   - Correctness passed focused packed/framegroup parity.
   - Isolated interleaved timing artifact:
     `research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_packed_predecode_variant_interleaved_timing_probe_tracks1024_prewarm_warm3_steps8_64_128.json`
   - Broad train/eval guard was stopped after the 64f packed row because it was
     clearly worse: predecode packed 64f was about `282.879 ms` total /
     `258.017 ms` backward versus i16x3 64f about `6.000 ms` total /
     `5.422 ms` backward in that run.
   - Interpretation: saving bit-unpack work is not worth the extra threadgroup
     memory/state. The 128f packed loss is not just redundant decode.

2. Packed-loop sentinel fast path:
   - Added direct handling for `left_cut/right_cut == -1/-2` before falling back
     to `wf2_endpoint_record_coeff16_cut_depth`.
   - Correctness passed focused framegroup parity.
   - Isolated interleaved timing artifact:
     `research_experiments/world_foam_lane2/results/2026-05-16_delta_framegroup_packed_sentinelfast_variant_interleaved_timing_probe_tracks1024_prewarm_warm3_steps8_64_128.json`
   - The isolated probe was noisy: packed 64f mean/median looked good
     (`3.679 / 3.566 ms`), but packed 128f mean was still bad
     (`115.123 ms`) despite a reasonable median (`4.429 ms`).
   - The broad train/eval guard was stopped during the packed 128f row because
     it ran too long. Before stopping, the 64f row was mixed and not promotable:
     i16x3 `146.875 / 137.995 ms` total/backward, packed
     `183.897 / 95.938 ms` total/backward.
   - Interpretation: sentinel fast path is at best marginal and cadence-noisy;
     it did not produce a clean 128f fix.

Current state after this note:
- Both micro-edits were reverted from the shader source.
- Focused framegroup parity after revert:
  `PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_matches_scalar research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk -v`
  passed, 2 tests OK.
- The best next shader fork should avoid extra threadgroup state. Better
  candidates are split-array i16 owner/cut records or a sorted-change binary
  search/fallback path, with the latter requiring a producer-side sortedness
  invariant.
