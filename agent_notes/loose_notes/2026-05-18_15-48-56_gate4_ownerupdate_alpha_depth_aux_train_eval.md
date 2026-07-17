# Gate4 owner-update alpha/depth aux train/eval gate

The owner-update train/eval artifact proved optimizer-loop integration, but it
still used RGB reconstruction loss only. That meant `loss.backward()` exercised
the owner-update autograd path with zero alpha/depth output adjoints. The
separate render/VJP smoke already covered RGBA/depth seeds, but the
optimizer-loop path needed its own non-RGB-adjoint gate.

Changed the train/eval harness:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py
```

New behavior:

- `_render` now returns RGB, alpha, and depth from the autograd wrapper.
- Optional `--alpha-aux-weight` and `--depth-aux-weight` add tiny auxiliary
  losses to the RGB MSE.
- On the first step, alpha/depth outputs retain grad and the artifact records
  `first_alpha_output_grad_abs_sum` and `first_depth_output_grad_abs_sum`.
- Timing stays in `step_summary`; losses now live in `loss_summary`.

Changed the verifier:

```text
research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py
research_experiments/world_foam_lane2/test_verify_gate4_affine_train_eval.py
```

New verifier flag:

```bash
--require-alpha-depth-aux-loss
```

It requires positive top-level aux weights, active row `loss_terms`, and
positive alpha/depth output gradient sums.

Commands:

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
  --alpha-aux-weight 0.01 \
  --depth-aux-weight 0.01 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16.json

PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16.json \
  --vjp-mode direct_atomic_grad_only_ownerupdate \
  --require-alpha-depth-aux-loss \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_ownerupdate_rgba_depth_aux_render32_site12_2_4_8_16_verifier.json
```

Verifier result:

- status: `ok`
- alpha/depth aux weights: `0.01 / 0.01`
- alpha output grad abs-sum 2/4/8/16:
  `0.0199991 / 0.0199991 / 0.0199991 / 0.0199991`
- depth output grad abs-sum 2/4/8/16:
  `0.0001728 / 0.0001624 / 0.0001567 / 0.0001569`
- train PSNR 2/4/8/16: `11.794 / 11.879 / 12.020 / 12.103`
- heldout PSNR 2/4/8/16: `12.038 / 13.058 / 13.130 / 13.274`
- total mean scale 2->16: `1.126x` for an `8x` frame-count increase
- backward mean scale 2->16: `0.949x`
- train/heldout mixed tape storage scale 2->16: `0.992x / 1.013x`
- explicit ray storage scale 2->16: `8.0x`

Gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_train_eval.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

The combined Gate4 suite ran 28 tests and passed.

Interpretation: this is the strongest current owner-update correctness gate.
It proves moving-camera optimizer-loop execution with nonzero RGB, alpha, and
depth output adjoints through the owner-update forward/backward path. It still
does not change the speed conclusion: owner-update is correctness-positive but
not the competitive runtime path.
