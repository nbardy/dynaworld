# STAR UVT Sparse-Pixel Target-Grid VJP

## Question

The current 64f/512px/8192-tube STAR UVT target-grid plus frozen RGB-probe
objective was still spending a large backward bucket even after the analytic
target-grid/probe VJP gate. The handoff suspicion was that the target-grid
loss creates a mostly-zero dense `[frames, feature_dim, height, width]`
gradient after trilinear downsampling, but the renderer backward still walks
every dense render pixel.

This session tested that directly.

## What Changed

- Added `direct_atomic_feature_sparse_pixels_backward` in
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`.
- Added C++/Torch registration:
  `direct_atomic_feature_sparse_pixels_backward_with_bins`.
- Added Python wrapper:
  `direct_atomic_feature_sparse_pixels_backward_cached_bins`.
- Added trainer mode:
  `feature_target.image_vjp_mode=analytic_sparse_pixels`.
- Added replay config:
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsepixvjp.jsonc`.
- Extended
  `research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py`
  so the same parity profiler can run `analytic`, `analytic_sparse_pixels`, and
  autograd comparisons.

The sparse kernel reuses forward bins, decodes each sparse pixel id into
`f/y/x`, sorts the tile's tubes locally in one thread, runs the same reverse
direct-atomic math, and only accumulates gradients for those sparse pixels.

## Validation

Build:

```bash
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld \
  python setup.py build_ext --inplace
```

Static:

```bash
.venv/bin/python -m py_compile \
  src/train/train_star_uvt_feature_overfit.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  third_party/fast-mac-gsplat/variants/star_uvt_v0/torch_gsplat_bridge_star_uvt/feature_rasterize.py
```

Profiler smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --image-vjp-mode analytic_sparse_pixels --warmup 0 --repeat 1 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_profile_smoke
```

Profiler repeat:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --image-vjp-mode analytic_sparse_pixels --warmup 1 --repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_sparse_pixel_vjp_profile
```

Dense analytic rerun:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/star_uvt_targetgrid_vjp_bridge_profile.py \
  --image-vjp-mode analytic --warmup 1 --repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_targetgrid_dense_analytic_vjp_profile_rerun
```

Trainer smoke:

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparsepixvjp.jsonc
```

## Results

Profiler repeat-3:

| path | pass | total | render fwd | image VJP | sparse pack | renderer bwd | param bwd | max grad err |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense analytic bridge | true | 1245.9ms | 523.3ms | 94.3ms | 0.0ms | 557.6ms | 26.9ms | 3.55e-08 |
| sparse-pixel bridge | true | 920.5ms | 525.9ms | 95.8ms | 184.0ms | 46.3ms | 25.6ms | 4.61e-08 |

Trainer 5-step:

| path | pass | mean step | no-first step | mean backward | no-first backward | sparse pack | end loss | end probe PSNR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| dense analytic | true | 1409.0ms | 1318.0ms | 608.4ms | 594.0ms | 0.0ms | 0.885009 | 21.984 |
| sparse-pixel analytic | true | 1052.1ms | 973.7ms | 283.1ms | 254.5ms | 205.7ms | 0.885009 | 21.984 |

Sparse support:

```text
65,536 sparse pixels per 64f/512 step
16,777,216 dense pixels per 64f/512 step
fraction = 0.00390625
```

The first chunk of the trainer trace pays warmup (`74.4ms` pack), then most
chunks settle around `5.5-11.2ms` sparse pack and `7.3-16.4ms` backward. The
step-level no-first numbers are the better comparison.

## Interpretation

This is the first current-objective STAR UVT feature speed gate where changing
renderer backward survives the trainer loop. The target-grid loss really is
sparse enough that dense pixel traversal is wasteful: sparse renderer backward
drops from about `557.6ms` to `46.3ms` in the parity profile.

The remaining cost is not the sparse Metal kernel. It is still the dense Torch
image VJP and sparse packing. The next best shader/adapter step is to generate
target-grid trilinear sparse pixel ids and weights directly, and then combine
the hidden64 frozen-probe VJP into sparse values without materializing a dense
`[2,32,512,512]` chunk gradient.

## Handoff

- Use `feature_target.image_vjp_mode=analytic_sparse_pixels` for the fastest
  current target-grid/frozen-probe diagnostic.
- Keep the mode opt-in; it changes the backward implementation but not the
  objective.
- Do not spend more time on reducer modes for this keeper objective until there
  is a native sparse target-grid/probe VJP or a true fixedbin/tile-slot kernel.
- The dense analytic mode remains the simpler parity control.
