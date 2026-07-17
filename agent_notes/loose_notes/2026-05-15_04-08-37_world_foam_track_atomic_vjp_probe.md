# World Foam track-atomic VJP probe

## Context

After `direct_atomic_grad_only` became the best fused-slab mixed VJP path, I
tested one more STAR-UVT-shaped idea: accumulate all frame contributions for one
track locally and issue one atomic add per trainable site field. The goal was to
reduce atomics across frame count without changing the affine slab math.

Implementation lives in the forked shader variant:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
```

New mode:

```text
direct_atomic_track
```

## What changed

- Added Metal kernel `wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_track_tensor`.
- Added C++/pybind op `fused_slab_affine_num32_den16_vjp_direct_atomic_track`.
- Added Python autograd mode `vjp_mode="direct_atomic_track"`.
- Extended the smoke harness and train/eval harness so the variant can be timed
  against reducer/direct/rgb-only/grad-only paths.

## Validation

Static gates passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py

git -C third_party/fast-mac-gsplat diff --check -- variants/world_foam_lane2_fused_slab_v0
```

Focused smoke:

```text
research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_direct_atomic_track_smoke_2f_render16_pertrack.json
```

Smoke result:

```text
max_grad_delta_vs_reduce = 0.000213623046875
max_grad_rel_delta_vs_reduce = 5.02575520048994e-07

render16 / 2f VJP timing:
reduce:    5.822 ms
direct:    1.667 ms
grad-only: 3.289 ms
rgb-only:  1.743 ms
track:     1.442 ms
```

The smoke made track-atomic look promising.

## Real train/eval result

Command:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --config src/train_configs/local_mac_powerfoam_metal_multicam_deepview_3cam_train2_test1_quaternion_height_sv_raytrace_32_smoke.jsonc \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --steps 5 \
  --warmup-steps 1 \
  --vjp-reduce-chunk-size 16 \
  --vjp-mode direct_atomic_track \
  --out-json research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_track_direct_atomic_render32_2_4_8_16.json
```

Track-atomic:

```text
frames   total ms   render ms   backward ms   train PSNR   heldout PSNR
2        7.354      2.678       2.761         11.794       12.038
4        7.296      2.701       2.963         11.879       13.058
8        9.336      3.243       4.183         12.020       13.130
16       13.476     3.997       6.752         12.103       13.274
```

Best prior `direct_atomic_grad_only`:

```text
frames   total ms   render ms   backward ms   train PSNR   heldout PSNR
2        7.173      2.574       3.042         11.794       12.038
4        7.626      2.616       3.269         11.879       13.058
8        7.948      2.659       3.721         12.020       13.130
16       9.316      2.717       5.299         12.103       13.274
```

## Takeaway

Track-atomic is correct and wins the tiny focused VJP smoke, but it is not the
real winner once placed inside the render32 train/eval loop. It helps backward
at 2f/4f, then loses total step time at 8f and badly at 16f. The likely cause is
the larger per-thread loop/body and replay pressure outweighing fewer atomics
as frame count grows.

The current recommended path remains:

```text
vjp_mode=direct_atomic_grad_only
```

World Foam is now practically sublinear over this smoke range, but not STAR-UVT
flat. The next meaningful improvement probably needs structural owner/candidate
reuse or stronger tube/time factorization, not another local alpha/RGB/depth
gradient micro-specialization.
