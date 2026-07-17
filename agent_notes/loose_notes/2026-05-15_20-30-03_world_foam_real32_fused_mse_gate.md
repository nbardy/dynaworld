# WorldFoam real-32f fused-MSE gate

Context: after the cached-clear fused-MSE path looked flat on the repeated-frame
speed smoke, I tested whether the result survives a stricter non-repeated
fixture. This used an existing 32-frame multicam config and did not pass
`--repeat-loaded-frames`.

Standalone fused-MSE command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --config research_experiments/world_foam_lane2/results/fixed_step_speed_compare_inputs_directatomic/128px_32f_config.json --tape-mode endpoint-record-edit-block-coeff-fused-mse --optimizer-mode manual-vjp --frame-counts 16,32 --render-size 16 --site-count 4 --steps 8 --warmup-steps 3 --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff_fused_mse_real32_manualvjp_warm3_steps8_render16_16_32.json
```

Standalone result:

- status: `ok`
- `allow_repeat_loaded_frames=false`, `repeat_loaded_frames=false`
- synthetic moving rays active: origin velocity `(0.08, 0.0, 0.02)`,
  direction velocity `(0.02, 0.0, 0.0)`
- 16f loaded 16f: total `3.455 ms`, fused/backward `2.847 ms`
- 32f loaded 32f: total `2.619 ms`, fused/backward `2.093 ms`
- 16f->32f total scale: `0.758x` for `2x` frames
- selected tape storage scale: `1.434x` for `2x` frames

Paired current-process compare command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py --config research_experiments/world_foam_lane2/results/fixed_step_speed_compare_inputs_directatomic/128px_32f_config.json --optimizer-mode manual-vjp --frame-counts 16,32 --render-size 16 --site-count 4 --steps 8 --warmup-steps 3 --include-block-coeff --include-block-coeff-rgb --include-block-coeff-fused-mse --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_blockcoeff_rgb_cachedclear_fused_mse_real32_warm3_steps8_render16_16_32.json
```

Paired 16f totals:

- endpoint-run: `4.298 ms`
- raw endpoint-record edit: `4.338 ms`
- block-coeff: `3.754 ms`
- block-coeff-rgb: `3.443 ms`
- block-coeff-fused-mse: `1.927 ms`

Paired 32f totals:

- endpoint-run: `3.936 ms`
- raw endpoint-record edit: `4.995 ms`
- block-coeff: `4.006 ms`
- block-coeff-rgb: `3.703 ms`
- block-coeff-fused-mse: `2.570 ms`

Interpretation:

- The fused-MSE path now has a real loaded 16/32-frame moving-ray smoke, not only
  repeated-loaded scaling. It is faster than endpoint-run at both 16f and 32f in
  this paired render16/site4 gate while keeping matched heldout PSNR.
- The underlying endpoint-run selected segment count is still nearly linear
  (`1.993x` for `2x` frames), so the fused speed win is a practical fused-loss
  hot-loop result, not proof that WorldFoam's endpoint topology is structurally
  STAR-like.
- STAR UVT still has cleaner saved frame-count render/projection evidence. The
  current WorldFoam result is stronger than before, but still not a matched
  quality/capacity head-to-head or a main-trainer integration.
