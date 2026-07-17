# WorldFoam i16x3 Fused-MSE Replay Fork

Followed the separated-int16 fused-MSE fork with an interleaved int16-record
variant:

```text
endpoint-record-edit-block-coeff16-i16x3-fused-mse
```

The point was to test whether one `(owner,left,right)` short3-style metadata
stream would keep the storage win while reducing the six separate int16 streams
used by the previous sidecar. It adds a Metal helper/kernel, C++ dispatch,
Python wrapper/export, train/eval mode wiring, storage accounting, and parity
coverage against the unpacked coeff16 fused path.

Verification:

```text
build_ext --inplace: ok
py_compile train_eval_owner_run_tape.py/test_probe_endpoint_record_edit_replay.py/ops.py/__init__.py: ok
focused replay suite: Ran 11 tests OK
full world_foam_lane2 unittest discovery: Ran 41 tests OK
```

Real 16/32 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16x3_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

Rows:

- 16f total/backward/storage/PSNR: `2.156 ms / 1.752 ms / 0.1919x full / 14.7128`
- 32f total/backward/storage/PSNR: `3.229 ms / 2.262 ms / 0.1375x full / 14.7955`
- scale 16f to 32f: total `1.50x`, backward `1.29x`, storage `1.43x` for `2x` frames

Repeated-loaded 16/32/64/128 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16x3_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Rows:

- total ms: `2.950 / 3.990 / 5.320 / 4.985`
- backward ms: `2.150 / 2.980 / 3.896 / 4.317`
- selected storage versus full: `0.1805x / 0.1391x / 0.1174x / 0.1068x`
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`
- scale 16f to 128f: total `1.69x`, backward `2.01x`, selected storage `4.73x` for `8x` frames

Interpretation: i16x3 is technically sublinear in runtime and storage on both
the real 16/32 and repeated-loaded scale gates, but it is not the practical
winner. It is slower than separated-int16 on repeat-loaded scaling and slower
than unpacked coeff16 on the smooth runtime curve. The current answer is:
WorldFoam has sublinear representation potential and practical sublinear fused
smokes, but it is not STAR-clean because selected replay segments still scale
nearly linearly with frame count and exact owner/cut replay remains branchy and
metadata-heavy.
