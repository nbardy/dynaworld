# STAR UVT Native Target-Area Geometry Split

Date: 2026-05-19 23:33:40 +07

## Original Goal Being Served

The original working goal from the handoff was to turn the STAR UVT/feature
shader state into an executable full-day plan: benchmark all relevant renderers
at matched frame counts and resolutions, identify the real backward bottlenecks,
choose the fastest usable UVT STAR path for a single-video overfit, then scale
the chosen path toward the 300-clip dataset while preserving the feature
splatting and WorldFoam investigation lanes.

The active narrowed goal is to repeat and harden the STAR UVT fast feature-shader
plan docs, fill any missing implementation details, execute the plan gate by
gate, benchmark each step, and record progress logs in markdown.

## What Changed

Added a benchmark-only native target-area split mode:

- `target_area_feature_grad_only`: computes loss and feature gradients, and
  intentionally zeros ma/q/opacity gradients.
- Existing `target_area_skip_feature_grad`: computes loss and ma/q/opacity
  gradients, and intentionally zeros feature gradients.

The wrapper now accepts mode bits `0..3`, so the two diagnostic bits can coexist
without changing trainer defaults.

Files touched:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
- `research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py`

## Commands

Extension rebuild:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Tiny parity:

```bash
rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_feature_grad_only \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_featureonly_tiny_parity_h64.json
```

Matched native-only timing used:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode <target_area_star_only|target_area_feature_grad_only|target_area_skip_feature_grad> \
  --timing-frames 64 --timing-size <256|512> --timing-tubes 8192 \
  --timing-feature-dim 32 --tile-capacity 256 --grid-side 64 --patch-size 8 \
  --timing-repeat <5 for 256, 3 for 512> --skip-baseline \
  --out-json <artifact>
```

## Results

| Size | Mode | Total ms | Backward ms |
|---:|---|---:|---:|
| 256 | full star-only | 741.97 | 581.33 |
| 256 | feature-only | 713.30 | 548.20 |
| 256 | geometry-only | 713.90 | 547.30 |
| 512 | full star-only | 2404.42 | 1919.75 |
| 512 | feature-only rerun | 2608.24 | 2106.69 |
| 512 | geometry-only | 2692.44 | 2173.97 |

The earlier skip-feature report had a same-build but slightly different timing
window: 256px backward `594.86 -> 562.17ms`, 512px backward
`1918.63 -> 1854.34ms`. The new same-session split confirms the same direction
at 256px and shows that the 512 partial modes can be slower than full.

## Conclusion

The native target-area hidden64 backward is dominated by shared work, not a
single output-gradient atomic loop. Feature-only and geometry-only still replay
the target-area traversal and hidden64 VJP. They do not expose a large separable
half of the kernel.

Current state:

- Hidden64 native target-area remains the full-support visual-VJP speed baseline.
- Hidden32 is faster but quality-invalid.
- Skip-feature and feature-only modes are diagnostics only.
- The remaining STAR UVT speed work should target shared hidden64 recompute,
  compact scalar/prefix tape, or the visual objective/support shape.

Report:

`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_geometrysplit_gate.md`
