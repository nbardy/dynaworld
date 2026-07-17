# STAR UVT hidden sigmoid-MSE native gate

## Why

The native-prep handoff gate removed most of the Torch-side linear RGB prep tax,
but it still did not cover the selected hidden/frozen-probe visual objective.
This follow-up tested the obvious next implementation: move the hidden
`FeatureToColor` RGB/loss VJP into the STAR UVT Metal reverse traversal.

## What Changed

- Added `direct_hidden_sigmoid_mse_backward(...)` to the STAR UVT feature
  rasterize wrapper.
- Added a Torch op and Metal bridge:
  `direct_atomic_feature_hidden_sigmoid_mse_backward`.
- Added a Metal kernel that renders a pixel feature cache, runs
  hidden `W1,b1 -> GELU -> W2,b2 -> sigmoid`, computes alpha-composed mean RGB
  MSE gradients, and backpropagates STAR-only gradients through the same reverse
  traversal.
- Added benchmark modes:
  `hidden_sigmoid_mse_star_only` and
  `hidden_sigmoid_mse_star_only_reduce_vec4`.
- Replaced Metal `erf()` with an Abramowitz-Stegun approximation because runtime
  shader compile failed on this target with `use of undeclared identifier 'erf'`.

This is intentionally benchmark-only. It returns STAR gradients, not colorizer
parameter gradients, and it does not wire into the trainer.

## Commands

```bash
rtk .venv/bin/python -m py_compile \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py

( cd third_party/fast-mac-gsplat/variants/star_uvt_v0
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 \
  --backward-mode hidden_sigmoid_mse_star_only_reduce_vec4 \
  --hidden-dim 32 \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_hidden_sigmoid_mse_star_only_reduce_vec4_tiny_parity.json
```

Timing rows then used `64f/8192t/F32`, warmup 3, repeat 5.

## Results

All hidden rows pass F4/F32 tiny parity and have zero overflow. Max STAR-gradient
errors stay below `3.8e-08`.

| Mode | Hidden | Res | Forward | Backward | Total |
| --- | ---: | ---: | ---: | ---: | ---: |
| scalar | 32 | 128 | `42.80ms` | `274.74ms` | `317.54ms` |
| scalar | 32 | 256 | `149.66ms` | `461.23ms` | `610.90ms` |
| scalar | 32 | 512 | `1384.01ms` | `1165.37ms` | `2549.39ms` |
| vec4 reduce | 32 | 128 | `44.99ms` | `294.61ms` | `339.60ms` |
| vec4 reduce | 32 | 256 | `149.63ms` | `484.98ms` | `634.61ms` |
| scalar | 64 | 256 | `145.27ms` | `672.01ms` | `817.27ms` |

## Read

Correctness is green, but this is not the next speed route. The hidden native
gate shows that simply fusing the hidden MLP and RGB loss into the STAR
traversal still leaves too much dense per-pixel work. It also still depends on
the dense feature forward image, which is noisy and expensive at 512px.

The vec4 reducer is worse than scalar here, so the bottleneck is no longer the
same feature-gradient atomic path as the earlier gradcache/reducer gates.
H64 at 256px confirms the trainer-capacity shape is too expensive as a naive
fused dense hidden kernel.

Next work should target compact visual gradients or a visibility/prefix tape
that avoids dense `[T,H,W,F]` support, while keeping sparse-forward batched
target/probe VJP as the selected speed comparison surface.

Report:
`outputs/benchmarks/2026-05-19_star_uvt_hidden_sigmoid_mse_native_gate.md`
