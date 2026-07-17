# WorldFoam cached-clear fused-MSE pass

Context: continued the STAR-UVT-inspired WorldFoam lane by removing more
per-step Python and allocation overhead from the endpoint record/edit block
coefficient fused-MSE path.

Changes:

- Added a Metal clear kernel for the fused loss scalar and site-RGBA gradient
  buffer, replacing host-side `torch::zeros` allocation with `torch::empty`
  plus an on-device clear.
- Cached block-coeff config tensors on the owner-run tape so the hot training
  loop can call the raw fused op directly instead of rebuilding wrapper config
  tensors every step.
- Kept the wrapper fallback path for callers that do not have cached configs.

Key commands:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 && PYTHONDONTWRITEBYTECODE=1 uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_probe_endpoint_record_edit_replay.py' -q
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/train_eval_owner_run_tape.py --tape-mode endpoint-record-edit-block-coeff-fused-mse --optimizer-mode manual-vjp --frame-counts 16,32,64,128 --render-size 16 --site-count 4 --steps 8 --warmup-steps 3 --repeat-loaded-frames --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff_fused_mse_cachedclear_manualvjp_repeat_loaded_warm3_steps8_render16_16_32_64_128.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/world_foam_lane2/compare_endpoint_run_record_edit_train_eval.py --optimizer-mode manual-vjp --frame-counts 16,32 --render-size 16 --site-count 4 --steps 8 --warmup-steps 3 --repeat-loaded-frames --include-block-coeff --include-block-coeff-rgb --include-block-coeff-fused-mse --out-json research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_blockcoeff_rgb_cachedclear_fused_mse_repeat_loaded_warm3_steps8_render16_16_32.json
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Results:

- Focused replay tests passed: 7 tests.
- Full WorldFoam lane2 unittest pass succeeded: 36 tests.
- Fused repeated-frame 16/32/64/128 run stayed effectively flat:
  - 16f: total 2.511 ms, fused/backward 2.068 ms
  - 32f: total 2.779 ms, fused/backward 2.317 ms
  - 64f: total 2.488 ms, fused/backward 2.056 ms
  - 128f: total 2.551 ms, fused/backward 2.183 ms
  - 8x frame-count scale gave 1.016x total-step scale and 1.056x fused/backward scale.
- Integrated compare passed all acceptance keys:
  - 16f endpoint-run total 4.653 ms vs fused-MSE total 2.318 ms.
  - 32f endpoint-run total 4.085 ms vs fused-MSE total 2.277 ms.
  - Fused-MSE heldout PSNR matched the other block-coeff path in this synthetic harness.

Interpretation:

- The latest fused-MSE path shows practical sublinear behavior in the repeated
  loaded-frame harness. It is no longer just a theoretical tape-compression
  argument for this narrow case.
- This still is not a full claim that WorldFoam is competitive with STAR UVT.
  The current proof is a small synthetic, repeated-frame, render-size-16,
  site-count-4 harness. The next required gate is a real moving-camera/non-
  repeated fixture and a direct STAR UVT head-to-head at matched frame counts.
