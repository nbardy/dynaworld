# WorldFoam I16cols Framegroup Negative

Added a split-column i16 record fork for the delta-replace coeff16 framegroup
path:

```text
endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse
```

The hypothesis was that the current i16x3 kernel's 3-wide, 6-byte record stride
might be contributing to the larger-frame MPS timing noise. This fork keeps the
same logical 6 bytes per record but stores records as column-major i16 streams:

```text
[owners...][left_cuts...][right_cuts...]
```

Touched surfaces:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal launcher and op binding:
  `endpoint_record_delta_replace_coeff16_i16cols_framegroup16_mse_vjp_direct_atomic_rgb_only`
- train/eval mode:
  `endpoint-record-delta-replace-coeff16-i16cols-framegroup16-fused-mse`
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
- one-step wire smoke passed and produced finite gradients
- warmed render16/site4 16/32 train-eval smoke:
  `research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16cols_framegroup16_train_eval_warm_smoke_render16_site4_16_32.json`

Warm smoke numbers:

- 16f total/backward mean: `301.169 / 275.116 ms`
- 32f total/backward mean: `237.964 / 228.532 ms`
- selected storage bytes: `51984 / 51950`
- heldout PSNR: `15.6795 / 15.6861`

Interpretation: correctness is fine, but runtime is catastrophically worse than
the live row-reference/loss-reduced i16x3 path, which is low single-digit ms on
the same small render16 family. Column-major i16 streams do not fix the MPS
cadence issue; they likely make memory access less coalesced for this kernel.
Keep the fork as a negative reference only. Do not promote it or spend 64/128
render32 time on it unless a later high-change-density fixture specifically
needs column-major records.
