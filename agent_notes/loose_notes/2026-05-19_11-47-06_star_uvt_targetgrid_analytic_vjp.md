# STAR UVT Target-Grid Analytic VJP

## Goal

The prior target-grid/frozen-probe bridge proved correctness but not speed: the
manual bridge still used Torch autograd to backpropagate through target-grid MSE
and the frozen hidden64 RGB probe to `grad_feature_image`. This follow-up tests
whether an analytic image-space VJP is worth porting forward.

## Implementation

Extended:

```text
research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py
```

New option:

```text
--image-vjp-mode analytic
```

The analytic path computes:

- target-grid MSE gradient directly on the downsampled `[T,F,H,W]` target grid
- frozen hidden64 RGB-probe VJP manually:
  `Conv1x1 -> GELU -> Conv1x1 -> sigmoid -> MSE`, including exact GELU
  derivative
- trilinear/nearest render-grid VJP through `aten.upsample_trilinear3d_backward`
  or `aten.upsample_nearest3d_backward`
- direct STAR UVT Metal feature backward with `grad_alpha=0`, because the
  target-grid/probe objective consumes the rendered feature image rather than
  the separate alpha output

Then it compares full model gradients against the normal
`render_uvt_feature_tubes_autograd(...).loss.backward()` path.

## Commands

Smoke:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --warmup 0 \
  --repeat 1 \
  --image-vjp-mode analytic \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_smoke
```

Repeat-5 gate:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --warmup 2 \
  --repeat 5 \
  --image-vjp-mode analytic \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_repeat5
```

## Results

Smoke:

```text
pass=true
autograd total 1456.0ms
analytic bridge total 1282.7ms
speedup 1.135x
loss max abs error 0
max grad abs error 1.55e-08
max tile/p95/cap 63/42/128
overflow 0
```

Repeat-5:

```text
pass=true
autograd total 1510.6ms
  render forward 571.1ms
  loss forward 73.3ms
  backward 866.3ms
analytic bridge total 1477.2ms
  render forward 563.3ms
  loss/VJP forward 96.1ms
  image VJP backward 110.2ms
  renderer backward 642.9ms
  param backward 64.6ms
speedup 1.023x
loss max abs error 0
max grad abs error 3.07e-08
max grad rel error 3.48e-05
max tile/p95/cap 63/42/128
overflow 0
unstable tiles 0
alpha grad missing chunks 32
```

Reports:

```text
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_smoke.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_smoke.json
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_repeat5.md
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_profile_64f512_from1300_repeat5.json
```

## Decision

This is a small but real next gate: the analytic VJP bridge is parity-clean and
repeat-positive. It does not replace the trainer yet, because this is still a
benchmark bridge that manually calls the renderer backward and then parameter
backward.

## Trainer Gate

The trainer now has an opt-in config key:

```text
feature_target.image_vjp_mode = "analytic"
```

Added smoke config:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_analyticvjp.jsonc
```

Runs:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_analyticvjp_64f512_from1300_5step.json
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_autogradvjp_64f512_from1300_5step.json
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_analyticvjp_64f512_from1300_5step_rerun.json
outputs/benchmarks/2026-05-19_star_uvt_targetgrid_analytic_vjp_trainer_report.md
```

Matched trainer result:

```text
autograd mean step 1303.6ms, no-first 1264.1ms, backward 693.8ms
analytic first mean step 1409.0ms, no-first 1318.0ms, backward 608.4ms
analytic rerun mean step 1304.6ms, no-first 1259.2ms, backward 590.4ms
all pass, all zero overflow, end losses match
```

The trainer result is a tie, not a promotion. The analytic path improves the
recorded backward bucket by `103.3ms` on the warm rerun, but shifts work into
the loss/VJP bucket, leaving mean step time essentially unchanged. Keep
`image_vjp_mode=analytic` as an opt-in diagnostic. The next speed path is either
a longer/fused native VJP that moves end-to-end step time or scalar
fixedbin/tile-slot feature-gradient accumulation.
