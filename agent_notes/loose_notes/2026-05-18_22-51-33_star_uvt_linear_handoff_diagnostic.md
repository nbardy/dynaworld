# STAR UVT Linear Handoff Diagnostic

Date: 2026-05-18

## Goal

Continue the STAR UVT feature-shader plan after the narrow
`fused_first3_sigmoid_mse` win. The question was whether the same idea stays
fast when generalized to a real linear colorizer:

```text
feature tubes -> feature_image [T,F,H,W] + alpha
logits = W @ feature_image + b
rgb = alpha * sigmoid(logits)
loss = mean((rgb - target_rgb)^2)
```

## Code Added

New benchmark/prototype API:

```text
torch_gsplat_bridge_star_uvt.feature_rasterize.direct_linear_sigmoid_mse_backward
```

It calls a new Metal op:

```text
star_uvt_v0::direct_atomic_feature_linear_sigmoid_mse_backward
```

Contract:

- inputs: STAR UVT feature tubes, `target_rgb [T,3,H,W]`,
  `color_weight [3,F]`, `color_bias [3]`
- output grads: `ma`, `q_uvt`, `opacity`, `feature`, `color_weight`,
  `color_bias`, and `tile_unstable`
- assumptions: zero RGB background, sigmoid linear colorizer, mean MSE, no
  hidden `FeatureToColor`, no pre-norm, no view conditioning
- cap: `feature_dim <= 64`

Diagnostic mode:

```text
linear_sigmoid_mse_skip_colorizer_grad
```

This uses the same colorizer weights for tube/feature gradients but leaves
colorizer weight/bias gradients at zero. It isolates whether the colorizer
parameter atomics were the obvious bottleneck.

## Commands

Build:

```bash
( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Parity:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --backward-mode linear_sigmoid_mse --skip-timing \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_tiny_parity.json
```

Timing:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode linear_sigmoid_mse --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_linear_sigmoid_mse_serial_64f_256_32768_f32.json
```

## Results

Parity:

- `linear_sigmoid_mse` passes F4/F32 tiny parity including
  `color_weight`/`color_bias` gradients.
- max errors are in the `1e-10` to `3e-08` range.
- `linear_sigmoid_mse_skip_colorizer_grad` passes tube/feature parity and
  returns zero colorizer grads as intended.

Target timing at `64f/256px/32768t/F32`, mean over 5 timed repeats:

```text
gradcache rerun after linear:
  634.0ms total / 156.5ms forward / 477.5ms backward

linear_sigmoid_mse:
  800.1ms total / 181.5ms forward / 618.5ms backward
  rerun: 792.9ms total / 177.4ms forward / 615.5ms backward

linear_sigmoid_mse_skip_colorizer_grad:
  956.0ms total / 241.9ms forward / 714.1ms backward
  rerun: 801.6ms total / 203.2ms forward / 598.5ms backward
```

## Interpretation

The generalized in-tile linear handoff is correct but not the next default.
It is slower than plain `gradcache` on the same target shape, even though the
narrow fixed-first3 handoff was fast.

The colorizer-gradient skip mode did not produce a stable mean win. That weakens
the simple theory that global `grad_color_weight` / `grad_color_bias` atomics
are the main regression. The heavier issue is likely the full generalized
per-pixel colorizer VJP inside the tile traversal plus timing noise from large
MPS launches. This needs a different shape, not a trainer promotion of the
current generalized kernel.

Current practical choice:

- Use `feature_direct_gradcache` for valid first-class training rows.
- Keep `feature_direct_fixedbin` as the overflow/promotion guard.
- Keep `fused_first3_sigmoid_mse` as evidence that a cheap RGB/logit handoff
  can work.
- Do not wire `direct_linear_sigmoid_mse_backward` into trainer configs as-is.

Next shader directions:

- image-space colorizer/loss reduction plus a cheaper RGB/logit-gradient
  renderer handoff
- optimized fixedbin feature backward
- two-pass/accum feature-gradient path that avoids the failed barrier-heavy
  per-contributor reduction shape

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
