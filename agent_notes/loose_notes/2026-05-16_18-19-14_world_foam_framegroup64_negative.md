# WorldFoam Framegroup64 Negative

Added a 64-frame i16x3 delta-replace coeff16 framegroup fork:

```text
endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse
```

The hypothesis was that the live row-reference/loss-reduced 32-frame chunk
kernel was still paying too many per-chunk loss/site-gradient atomics at 128f.
The 64-frame fork keeps the same row-reference i16x3 record layout but doubles
the frames per threadgroup, cutting chunk groups by half for long sequences.

Touched surfaces:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal launcher and op binding:
  `endpoint_record_delta_replace_coeff16_i16x3_framegroup64_mse_vjp_direct_atomic_rgb_only`
- train/eval mode:
  `endpoint-record-delta-replace-coeff16-i16x3-framegroup64-fused-mse`
- parity coverage in `test_probe_endpoint_record_edit_replay.py`

Verification:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_rowref_reduce_128_matches_scalar \
  research_experiments.world_foam_lane2.test_probe_endpoint_record_edit_replay.EndpointRecordEditReplayTests.test_delta_replace_framegroup_chunk_offsets_match_scalar_after_first_chunk -v

PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover \
  -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

Results:

- focused framegroup parity: 2 tests OK
- full world-foam lane: 88 tests OK
- render32/site12 16/32/64/128 smoke was killed during the 16f row after it
  failed to complete in the time the current 32-frame winner normally needs for
  the full sweep
- a tiny saved render16/site4 16f-only artifact completed:
  `research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_framegroup64_fused_mse_train_eval_16only_render16_site4_warm1_steps2.json`

Saved 16f-only numbers:

- total/backward mean: `158.978 / 134.583 ms`
- selected storage bytes: `51984`
- heldout PSNR: `13.7516`

Interpretation: correctness is fine, but the 64-thread framegroup is a runtime
negative. Doubling frames per threadgroup likely increases occupancy/register or
threadgroup-memory pressure more than it saves in global atomics. Do not promote
framegroup64. The current 32-frame row-reference/loss-reduced kernel remains
the practical winner in this branch.
