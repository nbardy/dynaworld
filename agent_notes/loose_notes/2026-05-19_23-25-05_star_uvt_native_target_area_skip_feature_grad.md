# STAR UVT Native Target-Area Skip Feature-Grad Diagnostic

Date: 2026-05-19 23:25 +07

## Question

After the hidden64 native target-area gate and the hidden32 rejection, the open
question was whether the remaining hidden64 backward cost was mostly the raw
feature-gradient atomic loop or the recompute needed to get geometry/opacity
gradients.

## Change

Added a benchmark-only `target_area_skip_feature_grad` mode to the native
target-area backward path. It keeps the hidden64 reverse math and keeps
`grad_dot_feature`, so geometry/opacity gradients still match. It only skips the
final atomic writes into `grad_feature`.

This is intentionally not a trainer promotion because STAR features would not
learn under this mode.

## Commands

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0
  rtk uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )

rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 --backward-mode target_area_skip_feature_grad \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_skip_feature_tiny_parity_h64.json

STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 --backward-mode target_area_star_only \
  --timing-frames 64 --timing-size 256 --timing-tubes 8192 --timing-feature-dim 32 \
  --tile-capacity 256 --grid-side 32 --patch-size 8 --timing-repeat 5 --skip-baseline \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_staronly_timing_64f256_8192t_h64_cap256_nativeonly_rerun.json

STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 --backward-mode target_area_skip_feature_grad \
  --timing-frames 64 --timing-size 256 --timing-tubes 8192 --timing-feature-dim 32 \
  --tile-capacity 256 --grid-side 32 --patch-size 8 --timing-repeat 5 --skip-baseline \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_skipfeature_timing_64f256_8192t_h64_cap256_nativeonly.json

STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 --backward-mode target_area_star_only \
  --timing-frames 64 --timing-size 512 --timing-tubes 8192 --timing-feature-dim 32 \
  --tile-capacity 256 --grid-side 64 --patch-size 8 --timing-repeat 3 --skip-baseline \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_staronly_timing_64f512_8192t_h64_cap256_nativeonly_rerun.json

STAR_UVT_TILE_CAPACITY=256 rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_hidden_target_area_kernel_benchmark.py \
  --feature-dims 32 --hidden-dim 64 --backward-mode target_area_skip_feature_grad \
  --timing-frames 64 --timing-size 512 --timing-tubes 8192 --timing-feature-dim 32 \
  --tile-capacity 256 --grid-side 64 --patch-size 8 --timing-repeat 3 --skip-baseline \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_sparse_hidden_target_area_skipfeature_timing_64f512_8192t_h64_cap256_nativeonly.json
```

## Results

Tiny H64 skip-feature parity passed for loss/ma/q/opacity. `grad_feature` was
zero by design.

| Resolution | Star-only backward | Skip-feature backward | Delta |
|---:|---:|---:|---:|
| 256 | `594.86ms` | `562.17ms` | `-32.69ms` / `-5.5%` |
| 512 | `1918.63ms` | `1854.34ms` | `-64.28ms` / `-3.4%` |

## Takeaway

Feature-gradient atomics are not the main hidden64 native target-area bottleneck.
They are a small win, but the big cost remains hidden/colorizer recompute plus
geometry/opacity reverse work per dense target-area pixel.

Next implementation should not spend a full gate on feature-atomic-only
accumulation for this path. Better candidates are hidden64 recompute reduction,
a visibility/prefix tape that avoids repeated colorizer work, or changing the
visual support/objective so dense full-cell8 reverse is no longer the target.
