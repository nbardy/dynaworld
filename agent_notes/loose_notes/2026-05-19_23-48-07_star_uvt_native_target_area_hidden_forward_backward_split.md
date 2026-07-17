# STAR UVT Native Target-Area Hidden Forward/Backward Split

Date: 2026-05-19 23:48:07 +07

## Why

Traversal-only proved hidden64 work was the largest removable native target-area
slice, but it did not say whether the cost was hidden forward/logits/sigmoid or
hidden backward/GELU/W^T.

## What Changed

Added a benchmark-only mode:

- `target_area_hidden_forward_only = 11`

This mode:

- skips feature-gradient atomics,
- skips geometry/opacity atomics,
- computes hidden64 forward/logits/sigmoid,
- skips hidden backward and feature-gradient cache construction.

It returns zero gradients by design and is not trainable.

Files changed:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_metal.mm`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
- `research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py`

## Commands

Rebuild:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
```

Tiny parity:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_hidden_forward_only \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_hiddenforwardonly_tiny_parity_h64.json
```

Timing:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_hidden_forward_only \
  --timing-frames 64 --timing-size <256|512> --timing-tubes 8192 \
  --timing-feature-dim 32 --tile-capacity 256 --grid-side 64 --patch-size <4|8> \
  --timing-repeat <5|3> --skip-baseline \
  --out-json <artifact>
```

## Results

| Size | Traversal-only | Hidden-forward-only | Recompute-only |
|---:|---:|---:|---:|
| 256 | `194.85ms` | `345.47ms` | `571.30ms` |
| 512 | `742.17ms` | `1192.78ms` | `2101.73ms` |

Approximate hidden forward slice:

- 256px: `150.62ms`
- 512px: `450.61ms`

Approximate hidden backward slice:

- 256px: `225.83ms`
- 512px: `908.96ms`

## Conclusion

The hidden backward side is larger than hidden forward. The next native
target-area speed pass should target GELU derivative plus W^T feature-gradient
accumulation, a structured/lower-rank quality-preserving decoder, or moving the
VJP boundary. Output-gradient masking is closed.

Report:

`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_forward_backward_split_gate.md`
