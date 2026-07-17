# STAR UVT Native Target-Area Traversal Floor

Date: 2026-05-19 23:44:21 +07

## Why

The recompute-only mode proved output-gradient atomics were not the bottleneck,
but still mixed tile/sample traversal, feature accumulation, hidden64 forward,
and hidden64 VJP. This gate split hidden64 work from the traversal floor.

## What Changed

Added a benchmark-only mode:

- `target_area_traversal_only = 7`

In the native kernel this forces:

- skip feature-gradient atomics,
- skip ma/q/opacity geometry-gradient atomics,
- skip hidden64 forward/VJP and sigmoid RGB path.

The mode still replays target-area samples and accumulates the pixel feature. It
returns zero gradients by design and is not trainable.

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
  --backward-mode target_area_traversal_only \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_traversalonly_tiny_parity_h64.json
```

Timing:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_traversal_only \
  --timing-frames 64 --timing-size <256|512> --timing-tubes 8192 \
  --timing-feature-dim 32 --tile-capacity 256 --grid-side 64 --patch-size <4|8> \
  --timing-repeat <5|3> --skip-baseline \
  --out-json <artifact>
```

## Results

| Size | Full backward | Recompute-only backward | Traversal-only backward |
|---:|---:|---:|---:|
| 256 | `581.33ms` | `571.30ms` | `194.85ms` |
| 512 | `1919.75ms` | `2101.73ms` | `742.17ms` |

The approximate hidden64 forward/VJP slice is `376.45ms` at 256px and
`1359.57ms` at 512px.

## Conclusion

The next target is now sharper: output-gradient atomics are closed, and
hidden64 forward/VJP is the biggest removable piece inside the native
target-area floor. Traversal plus feature accumulation is still substantial,
especially at 512px, but a gradient reducer alone cannot touch it.

Report:

`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_traversal_floor_gate.md`
