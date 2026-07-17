# WorldFoam Raw Edit Coeff16 Fused-MSE Fork

Added a new raw endpoint-record edit fused shader mode:

```text
endpoint-record-edit-coeff16-fused-mse
```

Motivation: the raw edit fused-MSE path already had the cleanest storage curve,
but it solved cut depths from boundary planes and per-sample rays inside the hot
shader. The block coeff16 fused path had the smoothest runtime, but its block
anchors made storage scale with frame count. This fork keeps the raw edit tape
and adds the block path's half-precision per-track linear depth coefficients.

Implementation:

- Metal kernel `wf2_endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Torch op `endpoint_record_edit_coeff16_mse_vjp_direct_atomic_rgb_only`
- Python wrapper/export and train/eval mode wiring
- storage accounting: `endpoint_record_edit.storage_bytes + coeff_f16 bytes`
- parity coverage inside the raw edit fused-MSE test, comparing loss and site
  gradients against the existing raw boundary/ray fused path

Verification:

```text
build_ext --inplace: ok
py_compile train_eval_owner_run_tape.py/test_probe_endpoint_record_edit_replay.py/ops.py/__init__.py: ok
focused replay suite: Ran 11 tests OK
full world_foam_lane2 unittest discovery: Ran 41 tests OK
git diff --check on touched files: ok
```

Real 16/32 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_coeff16_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

Rows:

- 16f total/backward/storage/PSNR: `2.574 ms / 1.564 ms / 0.1565x full / 14.7128`
- 32f total/backward/storage/PSNR: `3.002 ms / 2.066 ms / 0.0789x full / 14.7955`
- scale 16f to 32f: total `1.17x`, backward `1.32x`, selected storage `1.006x` for `2x` frames

Repeated-loaded 16/32/64/128 artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_coeff16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Rows:

- total ms: `2.530 / 10.603 / 4.169 / 3.413`
- backward ms: `1.948 / 6.647 / 3.280 / 2.714`
- selected storage bytes: `75420 / 75288 / 75300 / 75348`
- selected storage versus full: `0.1458x / 0.0728x / 0.0364x / 0.0182x`
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`
- scale 16f to 128f: total `1.35x`, backward `1.39x`, selected storage `0.999x` for `8x` frames

Follow-up 32-only reruns with warm5/steps12 were noisy: raw coeff16 measured
`69.7 ms` total / `50.6 ms` backward, while the known block coeff16 control
also slowed to `15.0 ms` total / `12.2 ms` backward. Treat those later reruns as
MPS/session noise evidence, not stable speed claims.

Interpretation: this fork is the best current WorldFoam storage answer and a
real 16/32 runtime improvement over raw edit fused-MSE. It is still not the
runtime winner because block coeff16 has the smoother 16/32/64/128 timing
curve. The current frontier is now clear: raw-edit coeff16 gives STAR-shaped
storage; block-coeff16 gives the smoothest fused runtime; a real fix would need
raw-edit storage plus bounded replay work, likely a sparse/keyed anchor scheme
instead of scanning raw edits from the base row or storing anchors every small
block.
