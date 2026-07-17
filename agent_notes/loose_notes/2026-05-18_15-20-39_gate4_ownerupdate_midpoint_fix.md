# Gate4 owner-update midpoint fix

## Problem

The Gate4 affine moving-camera bridge had owner-update explicitly out of scope.
The old owner-update artifact failed:

- forward owner-update RGB max error: about `0.424`
- owner-update grad-only VJP relative delta versus reduce: about `2.44e-4`

Rerunning the current Gate4 owner-update smoke before the fix reproduced the
same failure. Alpha/depth stayed near mixed precision tolerance, but RGB owner
assignment was wrong and the owner-update VJP was outside tolerance.

## Diagnosis

The owner-update kernels toggled the current owner at every candidate boundary:

```text
if owner == left -> right
else if owner == right -> left
```

That is not valid for the Gate4 slab CSR. The tape contains extra pair-boundary
candidates used to make the per-track slab conservative. Those boundaries are
not guaranteed to be lower-envelope owner transitions along the ray. The working
mixed forward/VJP path computes the owner at each segment midpoint, so it is
correct even with extra candidates. The toggle-owner shortcut can cross a
non-active pair boundary and assign RGB to the wrong site.

## Fix

Patched
`third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/csrc/metal/world_foam_lane2_shared_replay_tensor.metal`:

- `wf2_fused_slab_affine_num32_den16_ownerupdate_realray_rgba_depth_replay_tensor`
  now selects the owner at each segment midpoint.
- `wf2_fused_slab_affine_num32_den16_vjp_direct_atomic_grad_only_ownerupdate_tensor`
  now records segment owners from the same midpoint owner selection.

This makes the owner-update entrypoints correctness-first for the Gate4 affine
slab tape. It does not preserve the original toggle-owner speed shortcut; that
shortcut needs a true owner-transition tape.

## Build

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 &&
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

The C++ extension relinked. The Metal source is loaded dynamically from disk by
`world_foam_lane2_metal.mm`, so fresh Python smoke processes picked up the
edited `.metal` source.

## Verification

Quick 2-frame owner-update probe:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --tile-h 1 \
  --tile-w 1 \
  --include-vjp \
  --include-ownerupdate \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_render32_site12_2f_midowner_probe.json
```

Result: `status=ok`.

Full 2/4/8/16 owner-update artifact:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train:third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/smoke_fused_slab_affine_realray_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --layout per-track \
  --candidate-order slab-mid-depth \
  --tile-h 1 \
  --tile-w 1 \
  --include-vjp \
  --include-ownerupdate \
  --timing-iters 1 \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_render32_site12_2_4_8_16_midowner.json
```

Strict owner-update verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_render32_site12_2_4_8_16_midowner.json \
  --require-ownerupdate \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_ownerupdate_mps_vjp_render32_site12_2_4_8_16_midowner_verifier.json
```

Verifier result: `ok`.

Key numbers:

- owner-update forward max error: `0.00016689300537109375`
- owner-update VJP max grad delta versus reduce: `0.0859375`
- owner-update VJP max relative delta versus reduce:
  `7.990560968252593e-6`
- mixed max error: `0.00016689300537109375`
- mixed tape storage scale 2->16: `0.9667x`
- explicit ray storage scale 2->16: `8.0x`
- boundary-test ratios: `0.5 / 0.25 / 0.125 / 0.0625`

Python gates:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile \
  research_experiments/world_foam_lane2/verify_gate4_affine_tape_bridge.py \
  research_experiments/world_foam_lane2/test_verify_gate4_affine_tape_bridge.py

PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval -v
```

Result: 21 tests passed.

## Read

This closes the Gate4 owner-update correctness gap for affine slab tapes. It
also explains why the earlier owner-update idea was brittle: a boundary-id tape
is not enough to toggle owners unless the tape is specifically an exact owner
transition tape. For the current conservative slab candidate tape, midpoint
owner selection is the correct shader behavior.
