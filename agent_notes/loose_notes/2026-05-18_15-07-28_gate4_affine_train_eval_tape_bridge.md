# Gate4 affine train/eval tape bridge

## What changed

- Continued the Gate4 moving-camera World Foam lane after the reusable
  `Gate4AffineSlabTape` bridge had passed the focused render/VJP smoke.
- Moved the MPS train/eval harness in
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py`
  to build train and heldout tapes through `build_gate4_affine_slab_tape`
  instead of a private smoke-script CSR helper.
- Added storage and boundary-ratio fields to each train/eval row:
  `train_mixed_tape_storage_bytes`, `heldout_mixed_tape_storage_bytes`,
  `train_explicit_ray_storage_bytes`, `heldout_explicit_ray_storage_bytes`,
  `train_compiled_boundary_test_ratio`, and
  `heldout_compiled_boundary_test_ratio`.
- Added
  `research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py`
  and
  `research_experiments/world_foam_lane2/test_verify_gate4_affine_train_eval.py`.

## Verification

Syntax:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_train_eval.py \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py
```

Focused verifier tests:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

Result: 8 tests passed. The tests cover scoped acceptance, rejected scope
promotion, false acceptance flags, low PSNR, wrong boundary ratios,
non-sublinear timing, mixed tape storage growth, and non-linear explicit-ray
storage.

Combined Gate4 unit gate:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

Result: 18 tests passed.

MPS train/eval:

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
  --vjp-mode direct_atomic_grad_only \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_render32_site12_2_4_8_16.json
```

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_render32_site12_2_4_8_16.json \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_render32_site12_2_4_8_16_verifier.json
```

Verifier result: `ok`.

Key numbers:

- train PSNR at 2/4/8/16 frames:
  `11.794 / 11.879 / 12.020 / 12.103`
- heldout PSNR at 2/4/8/16 frames:
  `12.038 / 13.058 / 13.130 / 13.274`
- total mean time at 2/4/8/16 frames:
  `9.986 / 21.291 / 10.460 / 6.961 ms`
- backward mean time at 2/4/8/16 frames:
  `3.935 / 8.667 / 5.753 / 2.966 ms`
- total mean scale 2->16: `0.697x` for an `8x` frame-count increase
- backward mean scale 2->16: `0.754x`
- train mixed tape storage scale 2->16: `0.992x`
- heldout mixed tape storage scale 2->16: `1.013x`
- explicit ray storage scale 2->16: `8.0x` for train and heldout
- compiled boundary-test ratio:
  `0.5 / 0.25 / 0.125 / 0.0625`

## Interpretation

This is a positive scoped bridge result. The Gate4 affine moving-camera tape now
works not only in the render/VJP smoke but also inside a small optimizer loop
with heldout eval, nonzero gradients, parameter updates, finite outputs, and
PSNR above the verifier floor.

The storage result is the strongest evidence: mixed tape storage stays flat
while explicit ray storage scales linearly with frame count. That matches the
STAR-style thesis we are trying to port into World Foam.

Do not over-promote the timing result. The 4-frame row spiked badly, so this is
a smoke-scale optimizer-loop proof, not a stable benchmark. It is also still
frozen-geometry site-RGBA only: no full trainer claim, no full geometry-gradient
claim, and no STAR-UVT quality/capacity competitiveness claim.

## Next useful gate

Run a repeat or a stronger timed variant with more warmup/measured steps and a
matched STAR UVT reference under the same render/frame/site conditions. The
next claim to earn is not "can the Gate4 tape train at all"; it now can. The
next claim is whether the moving-camera World Foam path is repeatably fast
enough, and whether its PSNR/capacity remains acceptable once moved out of the
frozen site-RGBA micro setting.
