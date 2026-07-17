# Gate4 affine train/eval repeat20 timing gate

## Why

The first Gate4 affine moving-camera train/eval artifact proved that the
reusable tape object worked in an optimizer loop, but it used only one warmup
step and five measured steps. The 4-frame row spiked, so the end-to-end timing
was useful as a smoke but weak as a timing read.

## Changes

- Added `median_s` to each phase summary in
  `third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py`.
- Extended
  `research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py`
  with optional `--require-median-timing`, median first-to-last scale reporting,
  and row-level mean/median plus max/median spike guards.
- Extended
  `research_experiments/world_foam_lane2/test_verify_gate4_affine_train_eval.py`
  with missing-median and spiky-median rejection tests.

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

Result: 10 tests passed.

Combined Gate4 unit gate:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

Result: 20 tests passed.

## Artifact

Command:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 20 \
  --warmup-steps 5 \
  --vjp-mode direct_atomic_grad_only \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16.json
```

Verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16.json \
  --require-median-timing \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_gradonly_repeat20_render32_site12_2_4_8_16_verifier.json
```

Verifier result: `ok`.

Key numbers:

- train PSNR at 2/4/8/16 frames:
  `13.845 / 13.869 / 13.918 / 13.998`
- heldout PSNR at 2/4/8/16 frames:
  `14.288 / 14.504 / 14.536 / 14.592`
- total mean time at 2/4/8/16 frames:
  `4.696 / 4.559 / 3.700 / 4.816 ms`
- total median time at 2/4/8/16 frames:
  `4.757 / 4.456 / 3.550 / 4.578 ms`
- backward mean time at 2/4/8/16 frames:
  `2.085 / 1.921 / 1.698 / 2.318 ms`
- backward median time at 2/4/8/16 frames:
  `2.040 / 1.599 / 1.650 / 2.329 ms`
- total mean scale 2->16: `1.025x`
- total median scale 2->16: `0.962x`
- backward mean scale 2->16: `1.112x`
- backward median scale 2->16: `1.142x`
- train mixed tape storage scale 2->16: `0.992x`
- heldout mixed tape storage scale 2->16: `1.013x`
- explicit ray storage scale 2->16: `8.0x`

## Read

This is the strongest current Gate4 moving-camera train/eval timing read. It
removes the obvious 4-frame spike from the 5-step smoke and gives a median-based
sublinear optimizer-loop result for the reusable affine tape path. The result
supports the STAR-style storage and frame-scaling thesis at this narrow scope.

Do not promote it beyond that scope. It is frozen geometry and site-RGBA only,
not full geometry gradients, not the full trainer, and not a STAR-UVT
quality/capacity comparison.
