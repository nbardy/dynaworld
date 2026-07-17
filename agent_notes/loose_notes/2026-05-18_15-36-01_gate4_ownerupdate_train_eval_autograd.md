# Gate4 owner-update train/eval autograd

The fixed Gate4 owner-update shader had passed render/VJP smokes, including
the stronger RGBA/depth VJP seed gate. The remaining narrow gap was whether it
could run inside the frozen-geometry optimizer-loop harness rather than only as
a one-off VJP diagnostic.

Added a Python autograd wrapper in the WorldFoam Lane 2 fork:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py
```

New exported entrypoint:

```python
fused_slab_affine_num32_den16_ownerupdate_autograd(...)
```

It uses:

- forward: `fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay`
- backward: `fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate`

Then routed the train/eval harness mode:

```text
--vjp-mode direct_atomic_grad_only_ownerupdate
```

through owner-update forward and owner-update grad-only VJP. The harness now
ships `candidate_ids` and `boundary_site_pairs` to the MPS device bundle for
that mode.

Small 2-frame probe:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 2 \
  --warmup-steps 1 \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2f_probe.json
```

Probe result: status `ok`, gradients nonzero, loss decreased, parameters
updated, outputs finite.

Full 2/4/8/16 5-step owner-update train/eval:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 5 \
  --warmup-steps 1 \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16.json
```

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16.json \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_gradonly_render32_site12_2_4_8_16_verifier.json
```

Verifier result:

- status: `ok`
- train PSNR 2/4/8/16: `11.794 / 11.879 / 12.020 / 12.103`
- heldout PSNR 2/4/8/16: `12.038 / 13.058 / 13.130 / 13.274`
- total mean ms 2/4/8/16:
  `64.501 / 92.000 / 84.809 / 108.908`
- backward mean ms 2/4/8/16:
  `29.684 / 40.167 / 33.580 / 52.514`
- total mean scale 2->16: `1.688x` for an `8x` frame-count increase
- backward mean scale 2->16: `1.769x`
- train/heldout mixed tape storage scale 2->16: `0.992x / 1.013x`
- explicit ray storage scale 2->16: `8.0x`

Regression gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/ops.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/torch_world_foam_lane2_fused_slab/__init__.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

The combined Gate4 suite ran 23 tests and passed.

Interpretation: the owner-update shader is now optimizer-loop correct, not just
VJP-smoke correct. It is not the speed path. The midpoint owner selection fix
keeps it correct for extra candidate pair boundaries, but the midpoint owner
recompute makes it far slower than the normal `direct_atomic_grad_only`
repeat20 path. Use this as a correctness gate and keep the regular grad-only
path as the current practical runtime result.
