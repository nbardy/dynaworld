# World Foam grad-only direct-atomic VJP variant

Follow-up to the direct-atomic VJP work: the autograd backward only consumes
`grad_site_rgba`, but the first direct-atomic VJP still returned and wrote
RGB/alpha/depth replay outputs. I added a grad-only direct-atomic op and wired
it through the Metal/C++/Python/autograd stack.

Files touched in the fork:

- `csrc/metal/world_foam_lane2_shared_replay_tensor.metal`
- `csrc/metal/world_foam_lane2_metal.mm`
- `csrc/bindings.cpp`
- `torch_world_foam_lane2_fused_slab/ops.py`
- `torch_world_foam_lane2_fused_slab/__init__.py`
- `tools/smoke_fused_slab_affine_realray_mps.py`
- `tools/train_eval_fused_slab_mixed_mps.py`

New API:

- `fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only(...) -> grad_site_rgba`
- `fused_slab_affine_num32_den16_autograd(..., vjp_mode="direct_atomic_grad_only")`
- `tools/train_eval_fused_slab_mixed_mps.py --vjp-mode direct_atomic_grad_only`

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

Focused per-track VJP smoke after removing unused grad-only RGB accumulation:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_direct_atomic_grad_only_norgb_smoke_2f_render16_pertrack.json
status: ok
grad-only max delta vs reduce: 1.53e-4 absolute, 3.59e-7 relative
timings at render16/2f:
  reduce:      5.031 ms
  direct:      2.308 ms
  grad-only:   2.075 ms
```

Render32 kernel sweep before the small no-RGB cleanup:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_direct_atomic_grad_only_vjp_render32_pertrack_2_4_8_16.json
status: ok
frames:       2       4       8       16
forward ms:   1.753   2.219   2.106   2.966
reduce ms:    7.542   12.513  12.828  14.266
direct ms:    2.030   6.808   9.438   15.958
grad-only ms: 1.971   5.940   9.182   11.724
```

The grad-only VJP beat full direct atomics in this kernel sweep at all frame
counts, especially 16f. It also beat the reducer at all frame counts.

Best real train/eval artifact from the grad-only mode:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_gradonly_direct_atomic_render32_2_4_8_16.json
status: ok
frames:          2       4       8       16
total step ms:   7.173   7.626   7.948   9.316
render ms:       2.574   2.616   2.659   2.717
backward ms:     3.042   3.269   3.721   5.299
train PSNR:      11.794  11.879  12.020  12.103
heldout PSNR:    12.038  13.058  13.130  13.274
```

Post no-RGB cleanup train/eval rerun:

```text
artifact: research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_gradonly_norgb_render32_2_4_8_16.json
status: ok
frames:          2       4       8       16
total step ms:   7.756   9.362   9.187   10.650
backward ms:     3.368   3.386   4.680   5.906
```

The post-cleanup train/eval rerun was noisier/slower despite the focused smoke
improving, so cite both rather than pretending the latest single MPS run is
monotonic proof. The durable conclusion is that grad-only direct atomics are
correct and usually faster than the full-output direct-atomic VJP, but the
remaining scaling cost is still replay and atomic accumulation, not output
stores alone.

Current answer to the scaling question:

- World Foam is now practically sublinear over 2/4/8/16 frames in this fork.
- Best observed train step was 7.17 -> 9.32 ms from 2f -> 16f, about 1.30x for 8x frames.
- STAR UVT remains cleaner/flatter at larger frame counts because its tube/time
  structure avoids more per-frame replay.
