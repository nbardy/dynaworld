# WorldFoam Block-Coeff16 Fused-MSE Shader

Continued the WorldFoam fused-shader lane after the raw edit fused-MSE and fp32
block-coeff fused-MSE results. The useful next fork was not track-loop replay:
forward-only track-loop was already correct but slower from reduced GPU
parallelism. The better pressure point was block-coeff storage, because fp32
block-coeff fused-MSE already had the flat runtime curve but carried heavier
coefficient storage.

## What changed

- Added Metal kernel:
  - `wf2_endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only_tensor`
- Added host/binding/Python op:
  - `endpoint_record_edit_block_coeff16_mse_vjp_direct_atomic_rgb_only`
- Added train/eval tape mode:
  - `endpoint-record-edit-block-coeff16-fused-mse`
- Reused existing block edit anchors and `coeff_f16` boundary-depth helper.
- Extended replay tests so the coeff16 fused loss and gradient match the
  coeff16 forward plus RGB-only VJP path on a simple exact half-representable
  fixture.

## Gates run

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && \
  PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Build succeeded.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
```

Focused replay suite: `Ran 8 tests OK`.

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
  .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

Full lane suite: `Ran 38 tests OK`.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/train_eval_owner_run_tape.py \
  research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py
```

`py_compile` passed.

## Real 16/32 fixture

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

Results:

- 16f: total `2.0668 ms`, fused/backward `1.6765 ms`, selected storage
  `0.2677x` full, heldout PSNR `14.7128`.
- 32f: total `1.8928 ms`, fused/backward `1.5029 ms`, selected storage
  `0.2029x` full, heldout PSNR `14.7955`.
- 2x frame-count scale: total `0.916x`, backward `0.896x`.

## Repeated 16/32/64/128 scaling

Artifact:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff16_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Results:

- total ms: `2.752 / 2.476 / 2.241 / 2.759`
- fused/backward ms: `2.290 / 1.997 / 1.871 / 2.362`
- total scale: `1.003x` for `8x` frames
- backward scale: `1.031x` for `8x` frames
- selected storage versus full: `0.257x / 0.210x / 0.185x / 0.173x`
- selected storage scale: `5.38x` for `8x` frames

## Interpretation

This is the best practical runtime shape so far: the fused loss/VJP path is
near-flat from 16f to 128f and is faster than the older raw fused 128f path.
It is not a pure STAR-like storage result. Half coefficients only improve the
fp32 block-coeff storage ratio modestly because the block anchor/edit metadata
dominates the selected tape by 128f. Raw edit fused-MSE remains the cleaner
storage curve, but block-coeff16 fused-MSE is currently the cleaner runtime
curve.

Next good fork: pack/reduce block anchor metadata or move to a mixed raw-edit
plus sparse coefficient cache, rather than spending more time on track-loop
parallelism.
