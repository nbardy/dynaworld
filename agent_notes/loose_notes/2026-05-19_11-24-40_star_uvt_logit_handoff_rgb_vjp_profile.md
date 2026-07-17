# STAR UVT Logit-Handoff RGB-VJP Profile

## Goal

Test whether the new `logit_handoff_reduce_vec4` direct reducer can be used at a
trainer-style boundary instead of only as a synthetic direct-kernel row.

The narrow question was: for a linear no-pre-norm sigmoid `FeatureToColor` RGB
reconstruction loss, can we render feature tubes, compute the RGB/logit VJP in
image space, call `direct_logit_handoff_backward`, and then backpropagate the
returned tube gradients through the STAR parameter transforms with the same
model and colorizer gradients as the normal autograd wrapper?

## Implementation

Added:

```text
research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py
```

The script loads a first-class STAR feature config/checkpoint, requires a linear
sigmoid no-pre-norm colorizer (`colorize.hidden_dim=null`,
`colorize.pre_norm=false`, `colorize.activation=sigmoid`, no view conditioning),
and compares two paths on the same real video and checkpoint:

```text
autograd:
  render_uvt_feature_tubes_autograd(..., backward_mode=gradcache_reduce_feature_grad_vec4)
  colorize_and_compose(...)
  RGB MSE
  loss.backward()

handoff:
  render_uvt_feature_tubes(...)
  conv logits -> sigmoid -> alpha-composited RGB MSE
  analytic grad_logits, grad_alpha, colorizer weight/bias grads
  direct_logit_handoff_backward(..., backward_mode=logit_handoff_reduce_vec4)
  torch.autograd.backward((ma, q_uvt, opacity, feature), returned_grads)
```

The first implementation bug was a chunk coordinate double-shift: the profile
called `render_uvt_feature_tubes_autograd_frame_chunk` on inputs that had
already been shifted for a frame chunk. The fix was to call
`render_uvt_feature_tubes_autograd(*render_inputs, chunk_config, ...)` after
the shared `_chunk_inputs(...)` step, so the autograd baseline and handoff path
see the same chunk-local coordinates.

## Commands

Build after shader changes:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

8f/64px smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_8f_64_vjepa_target_gradcache_reduce_vec4_10step.jsonc \
  --warmup 0 \
  --repeat 1 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_8f64_smoke
```

64f/512px checkpoint profile:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  --config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_checkpoint_media.jsonc \
  --warmup 1 \
  --repeat 2 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_64f512_from1300
```

## Results

8f/64px smoke:

```text
pass=true
autograd total 78.8ms
handoff total 34.7ms
speedup 2.27x
loss max abs error 0
max tile/p95/cap 70/64/128
overflow 0
```

64f/512px/8192t from the 1300-step checkpoint:

```text
pass=true
autograd total 1691.0ms
  render forward 583.9ms
  colorize/loss forward 145.8ms
  backward 961.2ms
handoff total 1587.4ms
  render forward 581.6ms
  logit/loss VJP 339.0ms
  renderer backward 593.8ms
  param backward 72.9ms
speedup 1.065x
loss max abs error 0
max grad abs error 9.43e-09
max grad rel error 3.29e-05
max tile/p95/cap 63/42/128
overflow 0
unstable tiles 0
```

Reports:

```text
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_8f64_smoke.md
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_8f64_smoke.json
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_64f512_from1300.md
outputs/benchmarks/2026-05-19_star_uvt_logit_handoff_rgb_vjp_profile_64f512_from1300.json
```

## Decision

This is a positive parity gate and a small real-video speed win for linear RGB
reconstruction. It proves that the logit handoff can be made trainer-compatible
without losing gradients through the STAR parameter transforms.

It is not a promotion for the current V-JEPA target-grid/frozen-probe lane. The
current keeper objective has two pieces this profile does not cover:

- target-grid V-JEPA feature MSE, which produces arbitrary F-dimensional feature
  gradients rather than RGB-logit gradients
- the frozen hidden64 RGB probe, whose VJP is not the same as a linear
  `[3,F]` sigmoid colorizer

The next useful port is therefore a generic image-space VJP/native loss bridge
for the target-grid/frozen-probe objective, or an objective shape change that
can honestly use the linear logit handoff. Do not flip the main trainer to this
mode as a default just because this gate passes.
