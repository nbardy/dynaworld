# World Foam direct-atomic VJP and autograd forward fix

Context: after the fused mixed affine slab path (`num32_den16`) looked
sublinear in forward-only timing, the train/eval loop still paid too much
frame-count cost. The main surprise was that
`fused_slab_affine_num32_den16_autograd` was using the VJP reducer with zero
gradients in forward, so every training render paid backward-style work before
`loss.backward()`.

Changes in the fork:

- Added a direct-atomic VJP kernel for the mixed affine fused slab path,
  `fused_slab_affine_num32_den16_vjp_direct_atomic`, mirroring the STAR UVT
  idea of accumulating site RGBA gradients directly instead of materializing or
  chunk-reducing partials.
- Fixed the smoke harness so `--vjp-reduce-chunk-size` is actually passed into
  the reducer call. Earlier chunk-size sweep artifacts should not be used as
  chunk-size evidence.
- Changed `fused_slab_affine_num32_den16_autograd` so forward uses the pure
  replay kernel and backward chooses either `reduce` or `direct_atomic`.
- Added `--vjp-mode {reduce,direct_atomic}` to
  `tools/train_eval_fused_slab_mixed_mps.py`.

Validation:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py

git -C third_party/fast-mac-gsplat diff --check -- variants/world_foam_lane2_fused_slab_v0
```

Both passed.

Tiny render16 / 2f train smoke:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_forwardfix_smoke_2f_render16.json
status: ok
acceptance: gradients_nonzero, loss_decreased, outputs_are_finite, parameters_updated, zero_missing_sample_events
```

Kernel timing artifact:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_direct_atomic_vjp_render32_pertrack_2_4_8_16.json
status: ok
frames:                2       4       8       16
mixed forward ms:      2.373   2.179   2.971   3.493
VJP reduce ms:         5.802   6.528   11.733  13.796
VJP direct atomic ms:  4.739   6.013   11.145  16.370
direct/reduce speedup: 1.224   1.086   1.053   0.843
max mixed RGB error:   1.7e-4
max grad rel delta:    6.4e-6
```

Forward is clearly sublinear across frame count. Direct atomics help the VJP at
2/4/8f in this kernel timing artifact but lose at 16f, likely from MPS atomic
contention/noise.

Real train/eval after autograd forward fix, reducer chunk16:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_forwardfix_reduce_chunk16_render32_2_4_8_16.json
status: ok
frames:            2        4        8        16
total step ms:     10.150   9.441    10.723   12.777
render ms:         2.193    2.158    2.578    3.069
backward ms:       6.728    6.144    6.822    7.996
train PSNR:        11.794   11.879   12.020   12.103
heldout PSNR:      12.038   13.058   13.130   13.274
```

Real train/eval after autograd forward fix, direct atomic:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_forwardfix_direct_atomic_render32_2_4_8_16.json
status: ok
frames:            2        4        8        16
total step ms:     6.721    7.681    8.569    10.825
render ms:         2.538    2.533    2.351    3.252
backward ms:       2.714    3.542    4.219    5.796
train PSNR:        11.794   11.879   12.020   12.103
heldout PSNR:      12.038   13.058   13.130   13.274
```

Conclusion:

- In theory World Foam should amortize across frame count because the slab/candidate
  structure is shared and rays are affine in time.
- In practice, after the forward fix, this fork is sublinear in total step time
  over 2/4/8/16 frames, but not flat.
- STAR UVT remains cleaner/flatter because its time/tube factorization and
  direct-atomic path avoid more per-frame replay. World Foam still replays
  candidates for every frame in forward/backward, even though candidate storage
  and setup are shared.
- The next meaningful optimization is not another Python harness tweak; it is
  pushing more of the per-frame replay into STAR-like grouped/time-factored
  accumulation or reducing atomic contention in the direct VJP.
