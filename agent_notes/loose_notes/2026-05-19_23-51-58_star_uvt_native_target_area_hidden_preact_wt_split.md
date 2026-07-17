# STAR UVT Native Target-Area Hidden Preact/W^T Split

Date: 2026-05-19 23:51:58 +07

## Why

The hidden forward/backward split showed hidden backward was larger than hidden
forward. The remaining question was whether hidden backward was output-weight
and GELU derivative work, or the F32 `hidden_weight^T` feature-gradient matvec.

## What Changed

Added a benchmark-only mode:

- `target_area_hidden_preact_only = 19`

This mode:

- skips feature-gradient atomics,
- skips geometry/opacity atomics,
- computes hidden64 forward/logits/sigmoid,
- computes output-weight backprop plus GELU derivative to `grad_hidden_pre`,
- skips `hidden_weight^T @ grad_hidden_pre` feature-gradient reconstruction.

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
  --backward-mode target_area_hidden_preact_only \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_hiddenpreactonly_tiny_parity_h64.json
```

Timing:

```bash
STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 \
  --backward-mode target_area_hidden_preact_only \
  --timing-frames 64 --timing-size <256|512> --timing-tubes 8192 \
  --timing-feature-dim 32 --tile-capacity 256 --grid-side 64 --patch-size <4|8> \
  --timing-repeat <5|3> --skip-baseline \
  --out-json <artifact>
```

## Results

| Size | Traversal | Hidden forward | Hidden preact | Recompute |
|---:|---:|---:|---:|---:|
| 256 | `194.85ms` | `345.47ms` | `400.26ms` | `571.30ms` |
| 512 | `742.17ms` | `1192.78ms` | `1254.43ms` | `2101.73ms` |

Approximate slices:

- hidden forward/logits: `150.62ms` at 256px, `450.61ms` at 512px
- output+GELU prebackward: `54.79ms` at 256px, `61.65ms` at 512px
- F32 W^T feature-gradient matvec: `171.04ms` at 256px, `847.31ms` at 512px

## Conclusion

The expensive hidden-backward subpiece is the F32 `hidden_weight^T` matvec, not
the output-weight/GELU derivative. Future work should reduce or avoid that
channel-wise feature-gradient reconstruction before chasing more scalar
activation changes.

Report:

`outputs/benchmarks/2026-05-19_star_uvt_native_target_area_hidden_preact_wt_split_gate.md`
