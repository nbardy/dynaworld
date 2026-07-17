# WorldFoam Delta-Replace Coeff16 Fused MSE

Added a new endpoint-record tape mode:

```text
endpoint-record-delta-replace-coeff16-fused-mse
endpoint-record-delta-replace-coeff16-i16x3-fused-mse
```

The idea was to combine the raw-edit storage curve with less hot-shader replay:
pack full replacement endpoint rows only when a track changes, store coeff16
cut-depth data, and fuse RGB MSE plus site-RGBA VJP into one Metal dispatch.
The i16x3 follow-up stores replacement owner/left/right rows as interleaved
int16 triples to reduce record bytes further.

Touched surfaces:

- Metal kernel:
  `wf2_endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only_tensor`
- C++/Metal host wrapper and op binding:
  `endpoint_record_delta_replace_coeff16_mse_vjp_direct_atomic_rgb_only`
- Python wrapper/export in `torch_world_foam_lane2_fused_slab`
- train/eval mode, storage accounting, and final render support in
  `train_eval_owner_run_tape.py`
- changed-row parity coverage in `test_probe_endpoint_record_edit_replay.py`
  for both int32-row and i16x3-row delta replacement

Verification:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

Results:

- focused replay suite: 12 tests OK
- full lane suite: 42 tests OK
- repeat-loaded benchmark artifact:
  `research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replace_coeff16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json`
- rerun artifact:
  `research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replace_coeff16_fused_mse_repeat_loaded_rerun_warm3_steps8_render16_16_32_64_128.json`
- compact i16x3 artifact:
  `research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_delta_replace_coeff16_i16x3_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json`

First repeat-loaded 16/32/64/128 benchmark:

- total ms: `11.956 / 5.112 / 2.582 / 10.571`
- backward/fused ms: `8.715 / 2.981 / 1.826 / 8.715`
- selected storage bytes: `66580 / 66528 / 66528 / 66548`
- storage-vs-full: `0.1287x / 0.0644x / 0.0322x / 0.0161x`
- selected storage scale: `0.9995x` over `8x` frames
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`

Rerun after the kernel was already compiled:

- total ms: `3.398 / 5.903 / 6.409 / 5.634`
- backward/fused ms: `2.541 / 4.393 / 4.679 / 4.637`
- selected storage bytes: `66580 / 66528 / 66528 / 66548`
- storage-vs-full: `0.1287x / 0.0644x / 0.0322x / 0.0161x`
- selected storage scale: `0.9995x` over `8x` frames
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`

Compact i16x3 repeat-loaded 16/32/64/128 benchmark:

- total ms: `4.136 / 90.967 / 4.975 / 11.760`
- backward/fused ms: `2.660 / 72.771 / 3.670 / 9.015`
- selected storage bytes: `49936 / 49902 / 49902 / 49916`
- storage-vs-full: `0.0965x / 0.0483x / 0.0241x / 0.0121x`
- selected storage scale: `0.9996x` over `8x` frames
- heldout PSNR: `15.1983 / 15.1417 / 15.1905 / 15.1876`

Interpretation:

This is a storage win and runtime mixed/negative. It beats raw-edit coeff16
storage (`~66.5 KB` versus `~75.3 KB` on the repeat-loaded run), and the i16x3
follow-up improves that again to roughly `49.9 KB`. The storage curve is flat
across frame count. The rerun showed the very slow first 16f row was mostly
warmup/session noise, but the family still does not beat block-coeff16 on
smooth timing and generally trails raw-edit coeff16 except for that mode's
noisy 32f row. The i16x3 fork is even clearer: storage is best so far, but the
32f timing exploded (`90.967 ms` total, `72.771 ms` fused/backward), so it is
not the current runtime winner. Do not treat delta-replace coeff16 as the
winner; use it as evidence that better storage is possible, while the next fork
should target runtime directly.

One failed real 16/32 attempt stopped at 32f because the current fixture loaded
only 16 real frames without `--repeat-loaded-frames`; the synthetic repeat
artifact above is the valid frame-scaling comparison for 16/32/64/128.
