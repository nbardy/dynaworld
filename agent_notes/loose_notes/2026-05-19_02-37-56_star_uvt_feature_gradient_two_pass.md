# STAR UVT feature-gradient-only / two-pass diagnostic

Date: 2026-05-19

## Goal

Continue the STAR UVT fast feature-shader plan by testing the next obvious
split: separate geometry/opacity gradients from feature gradients, then measure
whether a two-kernel recompute can beat the current full `gradcache` backward.

This is part of the larger goal: make STAR UVT F32 feature tubes fast enough to
use for 512px source-view overfit and then scale the selected path to larger
video datasets without falling back to projected feature splatting.

## Current State Before This Gate

- `feature_direct_gradcache` is the current valid promoted STAR UVT feature
  renderer mode. It is correct and trainable, but only a modest speed win.
- `gradcache_skip_feature_grad` proved the feature-gradient atomics are a major
  target, but it intentionally returns wrong/zero feature gradients and cannot
  be used for training.
- Cached forward-bin reuse is a correct sidecar diagnostic, but it is mixed:
  it helps an isolated synthetic row and does not improve the first-class
  512px trainer row.
- First-class backward breakdown says 512px is not renderer-only:
  `FeatureToColor`/loss VJP is most of backward at 512px, while renderer
  backward is still material at 256px/32768t.

## Implementation

Added benchmark-only modes:

- `direct_atomic_feature_grad_only`
- `gradcache_feature_grad_only`
- `gradcache_two_pass_feature_grad`

The Metal bit layout now includes feature-gradient-only as bit `32` in
`direct_atomic_feature_backward`. In this mode the kernel:

- skips the `grad_dot_feature` loop used by geometry/opacity gradients
- skips `d_alpha`, `dT_next`, geometry, and opacity gradient accumulation
- keeps only the per-channel `grad_feature` atomics

The benchmark-level two-pass mode composes:

1. `gradcache_skip_feature_grad` for geometry/opacity
2. `gradcache_feature_grad_only` for features

The Python benchmark merges the two outputs and checks parity against the
autograd reference. This is deliberately not wired into the trainer.

Files changed:

- `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py`
- `research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py`
- `research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py`

Harness hardening:

- `direct_feature_mode_matrix.py` now deletes stale case JSON before each
  subprocess, so a failed case cannot keep old timings.
- The matrix now gives each subprocess an artifact-local `TMPDIR`, because the
  first refreshed row hit a PyTorch import failure when Python could not find a
  usable temp directory.

## Parity

Tiny feature-only parity:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache_feature_grad_only --feature-dims 4,32 --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_feature_grad_only_tiny_parity.json
```

Result: pass.

- F4 feature max error: `1.19e-6`
- F32 feature max error: `1.43e-6`

Tiny two-pass parity:

```bash
PYTHONPATH=third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --backward-mode gradcache_two_pass_feature_grad --feature-dims 4,32 --skip-timing \
  --out-json outputs/benchmarks/2026-05-19_star_uvt_feature_direct_metal_gradcache_two_pass_feature_grad_tiny_parity.json
```

Result: pass.

- F4 feature / ma / opacity / q max errors:
  `7.15e-7`, `1.19e-7`, `1.19e-6`, `4.17e-7`
- F32 feature / ma / opacity / q max errors:
  `1.91e-6`, `1.19e-6`, `1.53e-5`, `2.32e-6`

## Timing Matrix

Command:

```bash
TMPDIR=/Users/nicholasbardy/git/gsplats_browser/dynaworld/.tmp \
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --modes gradcache,gradcache_skip_feature_grad,gradcache_feature_grad_only,gradcache_two_pass_feature_grad \
  --sizes 256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 2 --repeat 5 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_direct_two_pass_feature_grad_matrix_256_512_64f_32768t_f32
```

Clean artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_direct_two_pass_feature_grad_matrix_256_512_64f_32768t_f32/summary.md
```

All 8 refreshed rows pass.

| size | mode | total ms | forward ms | backward ms |
| --- | --- | ---: | ---: | ---: |
| 256 | gradcache | 972.2 | 280.3 | 691.9 |
| 256 | gradcache_skip_feature_grad | 815.7 | 301.4 | 514.3 |
| 256 | gradcache_feature_grad_only | 817.1 | 288.1 | 529.1 |
| 256 | gradcache_two_pass_feature_grad | 1342.6 | 279.5 | 1063.2 |
| 512 | gradcache | 2466.8 | 1087.7 | 1379.2 |
| 512 | gradcache_skip_feature_grad | 1807.1 | 925.1 | 882.0 |
| 512 | gradcache_feature_grad_only | 1873.6 | 1011.6 | 862.0 |
| 512 | gradcache_two_pass_feature_grad | 2471.4 | 858.3 | 1613.1 |

Ratios:

- 256 two-pass / full gradcache: `1.381x` total, `1.536x` backward.
- 512 two-pass / full gradcache: `1.002x` total, `1.170x` backward.

The 512 total tie is not a win; backward is still slower and forward timing is
noisy.

Reverse-order 512 check:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_direct_two_pass_feature_grad_512_reverse_order/summary.md
```

- two-pass: `3216.7ms` total / `1821.1ms` backward
- gradcache: `2066.3ms` total / `1203.8ms` backward

That confirms the naive split-recompute path is negative.

## Decision

Do not port naive two-pass split-recompute to first-class training.

The split is correct and useful for measurement, but doing two STAR reverse
traversals loses to one traversal. The phrase "two-pass" should now mean one of
these:

- a true fixedbin/tile-slot accumulator where the first pass materializes a
  compact contributor structure and the second pass reduces feature gradients
  without repeating the expensive traversal
- a native image-space VJP/handoff that avoids dense F32 image-gradient
  backprop and avoids per-pixel `W^T` over every feature channel

## Plan Forward

1. Keep `feature_direct_gradcache` as the valid promoted feature renderer.
2. Keep `gradcache_skip_feature_grad`, `gradcache_feature_grad_only`, and
   `gradcache_two_pass_feature_grad` as diagnostics only.
3. Prototype a true fixedbin/tile-slot feature-gradient accumulator, not a
   duplicate traversal.
4. In parallel, keep attacking the 512px whole-graph bottleneck:
   `FeatureToColor`/loss VJP or objective/target representation.
5. Do not claim STAR UVT feature 512px replacement until Gate 4 quality closes
   against RGB STAR and the 512px speed path is no longer dominated by dense
   colorize/loss backward.

## Validation

- `py_compile` passed for:
  - `research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py`
  - `research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py`
- Artifact sanity passed:
  - 8 rows
  - all `status=ok`
  - all `pass=True`
  - sizes `{256, 512}`
  - `two_pass_feature_grad=True` and `feature_grad_only=True` columns present
