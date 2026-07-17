# WorldFoam raw endpoint-edit fused-MSE fork

Context: after the block-coeff fused-MSE shader made the hot loop fast, the
remaining question was whether the much more compact raw endpoint-record edit
tape could get the same fused loss+VJP treatment. This ports the fused-MSE idea
from the block-coeff path back onto the raw edit representation.

Changes:

- Added `wf2_endpoint_record_edit_mse_vjp_direct_atomic_rgb_only_tensor`, a
  Metal kernel that replays raw endpoint-record edit rows, computes RGB MSE
  against a track-major target, and atomically accumulates the site-RGBA VJP.
- Added the matching Metal host wrapper, Torch registration, Python wrapper, and
  package export under `world_foam_lane2_fused_slab_v0`.
- Added train/eval tape mode `endpoint-record-edit-fused-mse` and compare flag
  `--include-edit-fused-mse`.
- Added focused parity coverage:
  `test_edit_fused_mse_vjp_matches_render_loss_and_manual_vjp`.

Validation:

```text
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest research_experiments/world_foam_lane2/test_probe_endpoint_record_edit_replay.py -v
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=research_experiments/world_foam_lane2:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py' -q
```

The focused replay suite ran 8 tests OK. The full WorldFoam lane suite ran 38
tests OK after adding the compare harness coverage.

Real-loaded 16/32 raw fused command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --config research_experiments/world_foam_lane2/results/fixed_step_speed_compare_inputs_directatomic/128px_32f_config.json --tape-mode endpoint-record-edit-fused-mse --optimizer-mode manual-vjp --frame-counts 16,32 --render-size 16 --site-count 4 --steps 8 --warmup-steps 3 --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

Result:

- 16f: total `2.228 ms`, fused/backward `1.807 ms`, storage `0.1076x` full,
  heldout PSNR `14.7128`
- 32f: total `2.206 ms`, fused/backward `1.841 ms`, storage `0.0544x` full,
  heldout PSNR `14.7956`
- 16f to 32f: total scale `0.990x`, fused/backward scale `1.019x`, selected
  edit storage scale `1.009x`, edit-op scale `1.014x`

Paired real-loaded compare:

```text
research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_rawfused_blockcoeff_fused_mse_real32_warm3_steps8_render16_16_32.json
```

16f total times: endpoint-run `4.590 ms`, raw edit `4.728 ms`,
raw-edit-fused-MSE `2.600 ms`, block-coeff-rgb `4.261 ms`,
block-coeff-fused-MSE `1.943 ms`. The compact raw fused path is `0.566x`
endpoint-run at 16f and `0.550x` raw edit. Block-coeff fused remains faster,
but uses the heavier block-coeff tape (`0.3166x` full storage versus `0.1076x`
for raw edit fused at 16f).

Repeated-loaded 16/32/64/128 raw fused command:

```text
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --tape-mode endpoint-record-edit-fused-mse --optimizer-mode manual-vjp --frame-counts 16,32,64,128 --render-size 16 --site-count 4 --steps 8 --warmup-steps 3 --repeat-loaded-frames --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_fused_mse_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
```

Result:

- total means `3.134/2.628/3.018/3.956 ms`
- fused/backward means `2.285/2.169/2.562/3.571 ms`
- `8x` frame-count scale gives `1.262x` total-step scale and `1.563x`
  fused/backward scale
- selected edit storage scale is `0.999x` and edit-op scale is `0.996x`

Interpretation:

- Raw endpoint-record edit fused-MSE is now the best compact-storage speed path.
  It preserves the STAR-shaped edit tape storage signal and makes that path
  speed-positive.
- It still is not the fastest fused hot loop: block-coeff fused-MSE wins on
  runtime at 16f/32f, but pays extra storage.
- This is fixed-geometry/site-RGBA, render16/site4, MPS smoke-scale evidence.
  It is not yet a main-trainer or matched STAR-UVT quality/capacity claim.
