# WorldFoam int16 Fused-MSE Metadata Fork

Continued the WorldFoam fused-shader fork after the half-coeff block fused-MSE
path produced the cleanest runtime curve but still carried block anchor/edit
metadata. The question was whether we could move closer to STAR UVT's compact
temporal representation without paying the packed-bitfield decode cost.

Added two sidecar interpretations of the block coeff16 fused-MSE tape:

- `endpoint-record-edit-block-coeff16-packed-fused-mse` packs
  `(owner,left,right)` into one int32 record for anchors and op payloads.
- `endpoint-record-edit-block-coeff16-i16-fused-mse` keeps separate owner,
  left, and right arrays, but stores them as int16.

The packed fork is a storage win but a runtime negative. Real 16/32 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_packed_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

It reports:

- 16f total/backward/storage/PSNR: `7.238 ms / 5.015 ms / 0.1666x full / 14.7128`
- 32f total/backward/storage/PSNR: `4.768 ms / 3.399 ms / 0.1157x full / 14.7955`

The int16 fork is the better tradeoff. It uses a new Metal helper that loads
block edit rows from int16 owner/left/right streams, a fused MSE/VJP kernel,
C++ dispatch, Python wrapper/export, train/eval mode wiring, storage
accounting, and a replace-op parity test against the unpacked coeff16 fused
path.

Verification:

```text
build_ext --inplace: ok
py_compile train_eval_owner_run_tape.py/test_probe_endpoint_record_edit_replay.py/ops.py/__init__.py: ok
focused replay suite: Ran 10 tests OK
full world_foam_lane2 unittest discovery: Ran 40 tests OK
git diff --check on touched WorldFoam files: ok
```

Real 16/32 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

Rows:

- 16f total/backward/storage/PSNR: `1.415 ms / 1.089 ms / 0.1919x full / 14.7128`
- 32f total/backward/storage/PSNR: `2.673 ms / 2.288 ms / 0.1375x full / 14.7955`

The real 16/32 artifact has `status=failed` only because strict backward
sublinearity fails: backward scale is `2.10x` for `2x` frames. Total step is
still slightly sublinear at `1.89x`, and all rows are otherwise `ok`.

Repeated-loaded 16/32/64/128 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_i16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Rows:

- total ms: `2.976 / 2.433 / 4.224 / 2.859`
- backward ms: `2.236 / 2.026 / 3.341 / 2.147`
- selected storage versus full: `0.1805x / 0.1391x / 0.1174x / 0.1068x`
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`
- scale 16f to 128f: total `0.961x`, backward `0.960x`, selected storage
  `4.73x` for `8x` frames

Interpretation: WorldFoam is sublinear in storage by construction when using
the endpoint record tape, and fused-MSE makes runtime sublinear on the
repeat-loaded scale gate. It is not yet STAR UVT-clean on real frame-count
changes. STAR UVT gets cleaner math because its temporal object is a fixed-size
basis/tube, while exact WorldFoam replay still carries branchy owner/cut edit
metadata and per-sample row reconstruction. The practical current winner is
still unpacked coeff16 fused-MSE for runtime smoothness; int16 fused-MSE is the
best storage/runtime compromise so far.
