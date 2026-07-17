# WorldFoam I16x4 And Binary-Search Negatives

Followed up the flat-storage delta-replace coeff16 result with two runtime
experiments:

```text
endpoint-record-delta-replace-coeff16-i16x4-fused-mse
```

and a temporary binary-search change-selection helper for the existing i16x3
kernel.

## I16x4 Fork

The hypothesis was that i16x3's 6-byte stride might be causing the timing
instability. I added a padded i16x4 mode that stores replacement
`owner,left,right,unused` records as 8-byte records while preserving the
delta-replace storage curve.

Touched surfaces:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal launcher and op binding:
  `endpoint_record_delta_replace_coeff16_i16x4_mse_vjp_direct_atomic_rgb_only`
- Python wrapper/export in `torch_world_foam_lane2_fused_slab`
- train/eval mode:
  `endpoint-record-delta-replace-coeff16-i16x4-fused-mse`
- changed-row parity coverage in `test_probe_endpoint_record_edit_replay.py`

Verification:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

Results:

- focused replay suite: 12 tests OK
- full lane suite: 42 tests OK

The first i16x4 implementation used a Metal `short4*` buffer signature. Its
16/32/64/128 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x4_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

- total ms: `29.012 / 12.648 / 5.052 / 11.262`
- backward/fused ms: `23.284 / 10.114 / 3.596 / 8.209`
- selected storage bytes: `55484 / 55444 / 55444 / 55460`
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`

I then changed the live i16x4 kernel to scalar `short*` loads with stride 4.
The 16/32 control still failed as a runtime idea:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x4_scalar_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_control.json
```

- total ms: `23.145 / 31.418`
- backward/fused ms: `18.352 / 24.052`
- selected storage bytes: `55484 / 55444`

Interpretation: padding from 6 to 8 bytes keeps the storage below int32 delta
replace but makes runtime worse in this MPS kernel. Do not promote i16x4 as a
runtime fix.

## Binary-Search Change Selection

I temporarily replaced the per-sample linear scan for the last changed frame
with a small binary-search helper in the delta-replace coeff16 kernels. Parity
passed, but timing was much worse, so I reverted the helper and restored the
linear scan in live code.

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-16_endpoint_record_delta_replace_coeff16_i16x3_binarysearch_fused_mse_repeat_loaded_warm5_steps12_render16_16_32_control.json
```

- status: `failed`
- total ms: `21.351 / 47.938`
- backward/fused ms: `13.947 / 35.825`
- selected storage bytes: `49936 / 49902`

Interpretation: the current probe has few enough change events per track that
branchier binary search is worse than the simple linear scan. Do not repeat this
without a separate high-change-density fixture.
