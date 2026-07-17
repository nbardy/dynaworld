# STAR UVT tile-slot reducer isolation gate

Date: 2026-05-19

## Goal

After the tile-slot budget showed that a compact scalar accumulator is the only
plausible feature-gradient direction, isolate the reducer that already exists
inside `direct_atomic_feature_backward`.

Question: does the tile-slot reducer itself reduce feature-gradient backward
time when geometry/opacity work is removed, and does the same mechanism still
help a full single-pass synthetic backward?

## Implementation

Added benchmark-mode aliases:

- `gradcache_feature_grad_only_reduce`
- `gradcache_feature_grad_only_reduce_vec4`
- `gradcache_two_pass_feature_grad_reduce`
- `gradcache_two_pass_feature_grad_reduce_vec4`

These reuse existing Metal bits:

- gradcache: bit `1`
- scalar reducer: bit `4`
- vec4 reducer: bit `16`
- feature-only: bit `32`

So the reducer-only modes do not add a new Metal kernel. They expose the
existing threadgroup tile-slot reducer under the benchmark-only
feature-gradient-only path.

Files changed:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`
- `research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py`
- `research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py`

## Tiny Parity

Commands:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache_feature_grad_only_reduce --feature-dims 4,32 --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_feature_grad_only_reduce_tiny_parity.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache_feature_grad_only_reduce_vec4 --feature-dims 4,32 --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_feature_grad_only_reduce_vec4_tiny_parity.json

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache_two_pass_feature_grad_reduce_vec4 --feature-dims 4,32 --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_two_pass_feature_grad_reduce_vec4_tiny_parity.json
```

All pass.

Feature max errors:

- scalar reducer feature-only: `1.19e-6` for F4 and F32
- vec4 reducer feature-only: `1.19e-6` for F4 and F32
- two-pass vec4 reducer: `1.19e-6` feature, plus geometry/opacity/q errors
  within the existing tiny tolerance

## Feature-Only / Two-Pass Timing

Command:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes gradcache,gradcache_skip_feature_grad,gradcache_feature_grad_only,gradcache_feature_grad_only_reduce,gradcache_feature_grad_only_reduce_vec4,gradcache_two_pass_feature_grad,gradcache_two_pass_feature_grad_reduce,gradcache_two_pass_feature_grad_reduce_vec4 \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_reduce_matrix_256_512_64f_32768t_f32
```

Artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_tile_slot_reduce_matrix_256_512_64f_32768t_f32/summary.md
```

All 16 rows pass.

| size | mode | total ms | backward ms |
| --- | --- | ---: | ---: |
| 256 | gradcache_feature_grad_only | 791.7 | 532.8 |
| 256 | gradcache_feature_grad_only_reduce | 794.9 | 522.9 |
| 256 | gradcache_feature_grad_only_reduce_vec4 | 744.3 | 449.9 |
| 256 | gradcache_two_pass_feature_grad | 1208.3 | 930.6 |
| 256 | gradcache_two_pass_feature_grad_reduce_vec4 | 1083.7 | 827.2 |
| 512 | gradcache_feature_grad_only | 1730.2 | 869.1 |
| 512 | gradcache_feature_grad_only_reduce | 1722.0 | 823.5 |
| 512 | gradcache_feature_grad_only_reduce_vec4 | 1657.2 | 774.8 |
| 512 | gradcache_two_pass_feature_grad | 2477.1 | 1597.9 |
| 512 | gradcache_two_pass_feature_grad_reduce_vec4 | 3442.7 | 1599.5 |

Ratios:

- 256 feature-only vec4 reducer / direct feature-only backward: `0.844x`
- 512 feature-only vec4 reducer / direct feature-only backward: `0.891x`
- 256 two-pass vec4 reducer / plain two-pass backward: `0.889x`
- 512 two-pass vec4 reducer / plain two-pass backward: `1.001x`

## Full-Gradient Refresh

Command:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes gradcache,gradcache_reduce_feature_grad,gradcache_reduce_feature_grad_vec4 \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_full_reduce_refresh_256_512_64f_32768t_f32
```

Artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_full_reduce_refresh_256_512_64f_32768t_f32/summary.md
```

All 6 rows pass.

| size | mode | total ms | backward ms |
| --- | --- | ---: | ---: |
| 256 | gradcache | 944.6 | 676.6 |
| 256 | gradcache_reduce_feature_grad | 1021.5 | 702.8 |
| 256 | gradcache_reduce_feature_grad_vec4 | 919.5 | 654.5 |
| 512 | gradcache | 2324.1 | 1284.2 |
| 512 | gradcache_reduce_feature_grad | 1985.3 | 1141.6 |
| 512 | gradcache_reduce_feature_grad_vec4 | 1983.4 | 1108.0 |

## Decision

Keep single-pass vec4 tile-slot reduction live.

This gate overturns the overly broad "reducer is negative" phrasing. The
barrier-heavy reducer is not enough as a two-pass composition, and the scalar
reducer can still be slower, but the vec4 tile-slot reducer is a real
feature-only win and a synthetic 512px full-gradient win.

Do not promote two-pass reducer composition: it duplicates STAR traversal and
does not beat single-pass full gradcache at the trainer-relevant boundary.

## Next Work

1. Rerun or add a first-class 512px trainer row for
   `feature_direct_gradcache_reduce_vec4` against same-session plain gradcache
   before promoting it.
2. Keep the scalar weight/prefix tape idea alive, but only if it avoids
   per-channel tapes and avoids per-slot prefix recompute.
3. Keep native image-space VJP/handoff as the parallel path, because 512px
   first-class backward is still mostly `FeatureToColor`/loss VJP.

## Validation

- `py_compile` passed for:
  - `direct_feature_kernel_benchmark.py`
  - `direct_feature_mode_matrix.py`
  - `feature_rasterize.py`
- Artifact sanity passed for:
  - 16-row reducer isolation matrix
  - 6-row full-gradient reducer refresh
