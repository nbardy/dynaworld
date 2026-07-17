# World Foam Mixed VJP and Train/Eval Probe

Context: the mixed `num32_den16` affine slab replay path had become the best
strict forward-only World Foam lane, but it was not usable from a training
step. The gap was the lack of a reduced VJP/autograd path for the mixed moving
ray coefficient storage.

What changed:

- Added a Metal reduced VJP kernel for mixed affine CSR replay:
  `wf2_fused_slab_affine_num32_den16_vjp_partial_reduce_tensor`.
- Wired the kernel through C++/Torch as
  `fused_slab_affine_num32_den16_vjp_reduce`.
- Added a Python autograd wrapper:
  `fused_slab_affine_num32_den16_autograd`.
- Extended the fused slab smoke with `--include-vjp` so forward matching,
  finite gradients, and VJP timing can be recorded in the normal artifact.
- Added `tools/train_eval_fused_slab_mixed_mps.py`, a compact train/eval
  harness for frozen-geometry site-RGBA optimization using the mixed VJP path.

Files touched:

- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_metal.mm`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/bindings.cpp`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py`
- `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py`

Verification:

- `py_compile` passed for `ops.py`, the fused smoke, and the new train/eval
  harness.
- `git diff --check` passed for the fused slab variant.
- Extension build passed:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

- Small VJP runtime smoke:
  `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_vjp_smoke_2f_render16.json`
  - status `ok`
  - `mixed_vjp_matches_explicit_realray: true`
  - `mixed_vjp_gradients_finite: true`
- One-off finite-difference check for site-RGBA gradients:
  - max absolute error `9.202957153320312e-05`
  - max relative error `0.0031435663689964055`
- Render32 VJP scaling artifact:
  `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_vjp_slabmid_site12_render32_pertrack_2_4_8_16.json`
  - status `ok`
  - mixed VJP forward max error `0.0003178119659423828`
  - mixed VJP gradients finite
- Autograd smoke:
  - loss `0.35571911931037903`
  - grad abs sum `1.1669814586639404`
  - grad abs max `0.21780966222286224`
  - raw mixed forward vs autograd forward RGB max diff `0.0`
- Train/eval artifacts:
  - `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_render32_2_4.json`
  - `research_experiments/world_foam_lane2/results/2026-05-15_fused_slab_mixed_train_eval_render32_8_16.json`
  - both status `ok`

Render32 VJP timing, 2/4/8/16 frames:

- mixed forward: `2.081`, `1.769`, `2.006`, `2.136` ms
- mixed VJP reduce: `4.629`, `3.753`, `5.512`, `9.238` ms

Train/eval timing and PSNR, render32, 5 measured steps after 1 warmup:

- 2f: total step `0.01216` s, render `0.00459` s, backward `0.00485` s,
  train PSNR `11.794`, heldout PSNR `12.038`
- 4f: total step `0.00982` s, render `0.00439` s, backward `0.00418` s,
  train PSNR `11.879`, heldout PSNR `13.058`
- 8f: total step `0.01332` s, render `0.00561` s, backward `0.00614` s,
  train PSNR `12.020`, heldout PSNR `13.130`
- 16f: total step `0.02076` s, render `0.00969` s, backward `0.00954` s,
  train PSNR `12.103`, heldout PSNR `13.274`

Interpretation:

- This closes the immediate "forward-only" gap for the mixed shader: there is
  now a tested site-RGBA autograd path for moving affine rays.
- The training step is sublinear over frame count in this small render32
  frozen-geometry harness, but it is still not STAR UVT flat and not a full
  PowerFoam training claim.
- This is still frozen geometry/topology: site positions, weights, boundaries,
  CSR structure, and ray tracks receive no gradients.
- The next hard gap is full integration with the real PowerFoam training path
  or a stronger renderer schedule that reduces the remaining candidate/segment
  replay constant.
