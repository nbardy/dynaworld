# WorldFoam fused-MSE scaling pass

Question driving this chunk: is WorldFoam only theoretically sublinear, while STAR UVT is practically sublinear? The answer after this pass is mixed: the record/edit and block-coeff WorldFoam representation is structurally sublinear in edit/storage, and the new fused loss+VJP train path now shows practical sublinear timing on the small repeated-frame smoke, but it is not yet a stable STAR UVT-competitive claim.

What changed:

- Added a Metal `endpoint_record_edit_block_coeff_mse_vjp_direct_atomic_rgb_only` path in the `world_foam_lane2_fused_slab_v0` fork. It replays block-coeff RGB, computes MSE against a track-major target, accumulates scalar loss, and atomically accumulates direct site RGBA grads in one shader dispatch.
- Exposed the op through the C++ binding and Python `torch_world_foam_lane2_fused_slab` wrapper.
- Added focused MPS parity coverage against `block_coeff_rgb_replay + manual MSE grad + block_coeff_rgb_only_vjp`.
- Added `endpoint-record-edit-block-coeff-fused-mse` to `train_eval_owner_run_tape.py`.
- Added fused-MSE support to the compare wrapper. The wrapper now permits zero render time, because fused mode has no separate render pass.

Correctness evidence:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest discover -s research_experiments/world_foam_lane2 -p 'test_*.py'
```

Passed: 36 tests.

Probe artifact:

`research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_replay_blockcoeff_fused_mse_probe_render16_16f.json`

- status: ok
- fused loss max abs error: `8.731149137020111e-11`
- fused grad relative error: `1.0179285037776499e-05`
- isolated probe timing at 16f/render16:
  - endpoint forward: `2.594 ms`
  - block-coeff RGB forward: `1.359 ms`
  - block-coeff RGB-only VJP: `1.309 ms`
  - fused block-coeff MSE loss+VJP: `1.589 ms`

Training/scaling artifacts:

`research_experiments/world_foam_lane2/results/2026-05-15_endpoint_record_edit_blockcoeff_fused_mse_manualvjp_repeat_loaded_warm2_steps5_render16_16_32.json`

- status: ok
- frame scale: `2.0`
- total-step scale: `0.813`
- fused backward scale: `0.804`
- selected tape storage scale: `1.536`
- final train PSNR: `15.03 -> 15.09`
- final heldout PSNR: `15.62 -> 15.67`

Integrated compare artifact:

`research_experiments/world_foam_lane2/results/2026-05-15_endpoint_run_vs_record_edit_blockcoeff_rgb_fused_mse_repeat_loaded_warm2_steps5_render16_16_32.json`

Mean total step time:

| mode | 16f ms | 32f ms | scale |
| --- | ---: | ---: | ---: |
| endpoint-run | 4.569 | 5.663 | 1.239 |
| endpoint-record-edit | 6.449 | 7.415 | 1.150 |
| block-coeff | 3.532 | 4.696 | 1.330 |
| block-coeff-rgb | 5.250 | 3.188 | 0.607 |
| block-coeff-fused-mse | 5.378 | 2.407 | 0.448 |

Read:

- Structurally, WorldFoam is sublinear on edit/storage: record/edit ops are flat or slightly down from 16f to 32f on this repeated-frame smoke, and block-coeff storage grows `1.536x` rather than `2.0x`.
- Practically, this fused path can be sublinear in measured step time on the small smoke and is fastest at 32f in the integrated compare.
- It is not uniformly faster at 16f. The fused mode was slower than endpoint-run and block-coeff at 16f in the integrated compare, even though a separate warmed fused-only run measured `3.47 -> 2.82 ms`.
- MPS timing noise is material at these small timings. The next claim needs repeated runs and a larger real frame sweep before calling WorldFoam competitive with STAR UVT.

Current conclusion:

STAR UVT remains cleaner because the shader math is a flatter temporal table/tile contract. WorldFoam now has a real fused-loss train path that demonstrates the intended sublinear behavior on the smoke, but the implementation still pays for branchy edit replay, coefficient lookup, atomics, and small-kernel timing noise.
