# STAR UVT Logit Handoff Negative Gate

Date: 2026-05-18

## Goal

Follow the generalized linear handoff negative result with a cheaper
RGB/logit-gradient handoff. The idea was to avoid doing the full colorizer and
loss reduction inside STAR tile traversal:

```text
image space:
  logits = W @ feature_image + b
  rgb = alpha * sigmoid(logits)
  grad_logits = d loss / d logits
  grad_alpha = d loss / d alpha

Metal STAR backward:
  grad_feature_image = W^T @ grad_logits
  apply normal feature-tube reverse traversal
```

## Code Added

New benchmark/prototype API:

```text
torch_gsplat_bridge_star_uvt.feature_rasterize.direct_logit_handoff_backward
```

New Metal op:

```text
star_uvt_v0::direct_atomic_feature_logit_handoff_backward
```

Contract:

- inputs: STAR UVT feature tubes, `grad_logits [T,3,H,W]`,
  `grad_alpha [T,H,W]`, `color_weight [3,F]`
- output grads: `ma`, `q_uvt`, `opacity`, `feature`, and `tile_unstable`
- assumptions: image-space owns colorizer/loss differentiation and colorizer
  parameter gradients; the Metal op is renderer VJP only
- cap: `feature_dim <= 64`

## Commands

Parity:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --backward-mode logit_handoff --skip-timing \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_tiny_parity.json
```

Timing:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode logit_handoff --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_logit_handoff_serial_64f_256_32768_f32.json
```

Same-session comparison:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun6_after_logit_64f_256_32768_f32.json
```

## Results

Parity:

- `logit_handoff` passes F4/F32 tube/feature parity.
- max errors are in the `1e-10` to `3e-08` range.

Target timing at `64f/256px/32768t/F32`, mean over 5 timed repeats:

```text
logit_handoff:
  total: 835.63ms
  forward: 180.31ms
  handoff prep: 60.17ms
  renderer backward: 595.15ms

gradcache same-session rerun:
  total: 693.23ms
  forward: 164.27ms
  renderer backward: 528.96ms
```

## Interpretation

This is a correctness win and a speed loss. The logit handoff removes the dense
`[T,F,H,W]` gradient image input, but it still does `W^T @ grad_logits` per
pixel and still emits per-channel `grad_feature` atomics per contributor. The
extra image-space prep also costs about `60ms` in this shape.

Do not promote `direct_logit_handoff_backward` to trainer configs. Together
with the generalized in-tile linear handoff negative row, this narrows the next
real STAR UVT feature work to:

- optimized fixedbin feature backward
- two-pass/accum feature-gradient path
- a more radical handoff that avoids both full colorizer reduction and per-pixel
  `W^T` over all F channels

Current valid training default remains `feature_direct_gradcache`.

## Docs Updated

```text
research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md
research_experiments/star_uvt_feature_tubes/README.md
outputs/benchmarks/2026-05-18_renderer_scaling_report.md
outputs/benchmarks/2026-05-18_renderer_scaling_report.csv
EXPERIMENTS.md
TODO/README.md
PROJECT_INDEX.md
README.md
agent_notes/key_learnings.md
```
