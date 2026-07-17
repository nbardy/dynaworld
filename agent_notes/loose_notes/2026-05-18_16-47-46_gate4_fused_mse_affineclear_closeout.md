# Gate4 Fused-MSE Affine Clear Closeout

## Scope

We paused after the WorldFoam Gate4 affine slab fused-MSE path looked fast but
had suspicious 4f/8f quality. This note records the fix and the verification
state for the fork:

```text
third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
```

This is still a Gate4/frozen-geometry site-RGBA shader result, not a full
trainer, full geometry-gradient, or STAR-UVT quality claim.

## What changed

- Added the fused affine RGB-MSE VJP Metal kernel and Python/C++ binding:
  `fused_slab_affine_num32_den16_mse_vjp_direct_atomic_rgb_only`.
- Wired `vjp_mode=fused_mse_rgb_only` through
  `tools/train_eval_fused_slab_mixed_mps.py` and
  `tools/compare_fused_slab_vjp_modes_mps.py`.
- Added `_target_rgb_track_major(...)` so fused target layout matches the
  affine replay kernel's track-major `[track, frame, rgb]` order.
- Fixed the actual bug: the first fused wrapper reused
  `wf2_clear_endpoint_loss_site_rgba_grad_tensor`, but endpoint configs store
  `site_count` at `config_i32[3]`, while affine configs store it at
  `config_i32[2]`. For `site_count=12` and `frame_count=4/8`, that left
  some affine site-gradient rows uncleared and caused stale-gradient quality
  collapse. The fork now has and uses
  `wf2_clear_affine_loss_site_rgba_grad_tensor`.
- Expanded `tools/probe_fused_slab_affine_mse_vjp_mps.py` from a tiny 2-site
  probe to a 5-case parity probe: the original 2-site case plus 12-site cases
  at 2/4/8/16 frames. The 12-site cases touch all site rows and guard the
  affine clear layout.

## Verification

Build:

```bash
( cd third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
    python setup.py build_ext --inplace --force )
```

Expanded fused-MSE parity probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/probe_fused_slab_affine_mse_vjp_mps.py \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_fused_mse_vjp_multisite_parity_mps.json
```

Result:

```text
status ok
cases: tiny 2-site 2f plus 12-site 2f/4f/8f/16f
max loss diff: 3.725290298461914e-09
max grad diff: 0.0
all 12-site cases touched 12 sites
```

Focused Python tests:

```bash
PYTHONPATH=research_experiments/world_foam_lane2:src/train \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m unittest \
  research_experiments.world_foam_lane2.test_gate4_moving_ray_slab_compiler \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_tape_bridge \
  research_experiments.world_foam_lane2.test_verify_gate4_affine_train_eval \
  research_experiments.world_foam_lane2.test_train_eval_fused_slab_mixed_mps \
  research_experiments.world_foam_lane2.test_compare_fused_slab_vjp_modes_mps -v
```

Result: `Ran 35 tests ... OK`.

Fresh fused train/eval artifact:

```bash
PYTHONPATH=research_experiments/world_foam_lane2 \
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/world_foam_lane2_fused_slab_v0/tools/train_eval_fused_slab_mixed_mps.py \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --time-slabs 1 \
  --steps 20 \
  --warmup-steps 5 \
  --vjp-mode fused_mse_rgb_only \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_affineclear_repeat20_render32_site12_2_4_8_16.json
```

Formal verifier:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  research_experiments/world_foam_lane2/verify_gate4_affine_train_eval.py \
  --artifact research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_affineclear_repeat20_render32_site12_2_4_8_16.json \
  --frame-counts 2,4,8,16 \
  --render-size 32 \
  --site-count 12 \
  --vjp-mode fused_mse_rgb_only \
  --require-median-timing \
  --out-json research_experiments/world_foam_lane2/results/2026-05-18_gate4_affine_tape_train_eval_fusedmse_affineclear_repeat20_render32_site12_2_4_8_16_verifier.json
```

Result: `status ok`, `failures []`.

Verifier summary:

```text
frame scale: 8.0
total mean scale: 1.2346464561969395
total median scale: 1.4611021008013663
backward mean scale: 1.3620173235001605
backward median scale: 1.548935608644265
train mixed tape storage scale: 0.9917404592675542
heldout mixed tape storage scale: 1.0128231223285162
train explicit ray storage scale: 8.0
heldout explicit ray storage scale: 8.0
```

Fresh fused rows:

```text
F=2  total median 1.5219ms  backward median 1.2116ms  heldout PSNR 14.2880
F=4  total median 1.7047ms  backward median 1.3870ms  heldout PSNR 14.5043
F=8  total median 1.8926ms  backward median 1.5588ms  heldout PSNR 14.5355
F=16 total median 2.2236ms  backward median 1.8767ms  heldout PSNR 14.5921
```

Latest direct-vs-fused comparison artifact from the same fixed affine-clear
lineage:

```text
research_experiments/world_foam_lane2/results/2026-05-18_gate4_vjp_mode_compare_direct_vs_fusedmse_affineclear_repeat20_render32_site12_2_4_8_16.json
```

That compare showed fused total-step speedups over `direct_atomic_rgb_only` of
about `2.15x`, `1.93x`, `1.97x`, and `1.78x` at 2/4/8/16 frames, with heldout
PSNR differences at numerical noise and first-gradient abs-sum deltas around
`1e-6` or below.

## Interpretation

The fused-MSE shader is now a real local Gate4 win. It removes the separate
render/loss/autograd path for RGB MSE and keeps the warm-step timing strongly
sublinear over 2f to 16f at this small moving-camera affine slab gate.

This still does not make WorldFoam competitive with STAR UVT at the system
level. Tape build, candidate replay, and explicit ray storage remain separate
costs, and explicit rays still scale linearly with frame count. STAR UVT remains
cleaner because the time-tube reuse is native to the representation; WorldFoam
now has a useful STAR-like fused-loss/VJP idea, but it is still wrapped around
heavier bookkeeping.

## Next Gate

Do not spend the next pass on another micro-variant before a matched scale gate.
The useful next comparison is:

```text
STAR UVT direct_atomic vs WorldFoam Gate4 fused_mse_rgb_only
same render size / same frame counts / warm-step timing separated from tape build
```

Track at least:

- total warm-step median
- backward/fused-kernel median
- tape build wall time
- explicit ray storage
- mixed tape storage
- train/heldout PSNR
