# World Foam RGB-only direct-atomic VJP probe

Question: train/eval only uses RGB loss, so can backward skip the alpha/depth
adjoint math and beat the general grad-only direct-atomic VJP?

Change:

- Added `wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only_tensor`.
- Added C++/Python op `fused_slab_affine_num32_den16_vjp_direct_atomic_rgb_only`.
- Added autograd mode `vjp_mode="direct_atomic_rgb_only"`.
- Autograd only uses the RGB-only kernel when `grad_alpha is None` and
  `grad_depth is None`; if alpha/depth are used, it falls back to the general
  grad-only direct-atomic path.
- Added smoke-harness timing and reducer-parity diagnostics for the RGB-only
  variant.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py

git -C third_party/fast-mac-gsplat diff --check -- variants/world_foam_lane2_fused_slab_v0

( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

All passed.

Focused smoke:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_direct_atomic_rgb_only_smoke_2f_render16_pertrack.json
status: ok
rgb-only max delta vs reduce: 2.44e-4 absolute, 5.74e-7 relative
render16/2f timings:
  reduce:     5.838 ms
  direct:     2.087 ms
  grad-only:  2.318 ms
  rgb-only:   3.335 ms
```

Real train/eval:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_rgbonly_direct_atomic_render32_2_4_8_16.json
status: ok
frames:          2       4       8       16
total step ms:   8.145   8.320   11.365  11.024
render ms:       3.035   2.981   4.066   3.273
backward ms:     3.130   3.486   5.149   5.709
train PSNR:      11.794  11.879  12.020  12.103
heldout PSNR:    12.038  13.058  13.130  13.274
```

Conclusion:

- Correctness is fine, but RGB-only is not the recommended path.
- It did not beat the previous best grad-only direct-atomic train/eval artifact
  (`7.17/7.63/7.95/9.32 ms` total step for 2/4/8/16f).
- Likely reason: removing alpha/depth adjoint math was not the bottleneck;
  the expensive parts are still per-frame candidate replay, owner search,
  reverse transmittance recurrence, and atomics. The RGB-only specialization may
  also perturb Metal codegen/register allocation enough to lose despite doing
  fewer scalar operations.

Recommended current mode remains:

```text
vjp_mode = "direct_atomic_grad_only"
```
