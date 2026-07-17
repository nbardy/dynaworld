# STAR UVT Target-Grid VJP Bridge Profile

## Goal

Close the gap left by the linear RGB logit-handoff gate. The previous profile
proved a manual VJP bridge for linear no-pre-norm RGB reconstruction, but the
current keeper STAR feature objective is target-grid V-JEPA MSE plus a frozen
hidden64 RGB probe. This gate checks that actual objective.

## Implementation

Added:

```text
research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py
```

The script loads the same real 64f/512px/8192t STAR feature config/checkpoint
and compares:

```text
autograd:
  render_uvt_feature_tubes_autograd(..., backward_mode=gradcache_reduce_feature_grad_vec4)
  target-grid feature loss + frozen RGB-probe loss
  loss.backward()

image_vjp_bridge:
  render_uvt_feature_tubes(...)
  detach rendered feature image and compute target-grid/probe loss VJP in Torch
  direct_atomic_feature_backward(..., grad_feature_image, grad_alpha=0)
  torch.autograd.backward((ma, q_uvt, opacity, feature), returned_grads)
```

This is not a new Metal loss kernel. It is a correctness and timing bridge for
the current objective boundary.

## Commands

Smoke:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --warmup 0 \
  --repeat 1 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300_smoke
```

Repeat gate:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --warmup 1 \
  --repeat 2 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300
```

## Results

Smoke:

```text
pass=true
autograd total 1672.3ms
image_vjp_bridge total 1514.1ms
speedup 1.105x
loss max abs error 0
max grad abs error 4.80e-08
max tile/p95/cap 63/42/128
overflow 0
```

Repeat gate:

```text
pass=true
autograd total 1545.5ms
  render forward 574.1ms
  loss forward 78.3ms
  backward 893.1ms
image_vjp_bridge total 1594.3ms
  render forward 589.2ms
  loss forward 88.7ms
  image VJP backward 157.2ms
  renderer backward 690.4ms
  param backward 68.8ms
speedup 0.969x
loss max abs error 0
max grad abs error 2.57e-08
max grad rel error 2.91e-05
max tile/p95/cap 63/42/128
overflow 0
unstable tiles 0
alpha grad missing chunks 32
```

Reports:

```text
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300_smoke.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300_smoke.json
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_vjp_bridge_profile_64f512_from1300.json
```

## Decision

This is a positive correctness gate and a negative speed-promotion gate.

The bridge exactly answers the missing objective question: a manual image-space
VJP plus direct Metal feature backward matches normal autograd on the current
target-grid/frozen-probe objective. The gradient error is small enough to use
this as a future native-loss parity harness.

It does not make the row faster. The repeat gate is slightly slower than
autograd (`0.969x`). The next speed implementation should therefore either
fuse/simplify the target-grid/probe VJP itself or improve the renderer backward
with scalar fixedbin/tile-slot work. A Python-side manual bridge alone is not
the speed fix.
