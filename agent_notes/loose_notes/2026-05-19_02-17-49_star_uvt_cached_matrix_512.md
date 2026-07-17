# STAR UVT Cached-Bin 128/256/512 Direct-Mode Matrix

## Goal

Harden the STAR UVT feature direct-mode benchmark so cached-bin modes are part of
the reproducible matrix, then run the matrix through 512px to clarify whether
cached-bin reuse changes the current shader plan.

## Harness Change

Updated `research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py`:

- added cached-bin modes to the default allowed mode list
- added `kernel_backward_mode` and `cached_bins` columns to `summary.md` /
  `summary.csv`
- added a manifest note that cached-bin modes reuse forward bins and report the
  effective kernel mode

## Command

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --sizes 128,256,512 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 1 --repeat 3 --timeout-sec 600 \
  --out-dir outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached
```

## Results

Artifact:

```text
outputs/benchmarks/2026-05-19_star_uvt_feature_direct_mode_matrix_128_256_512_64f_32768t_f32_cached/summary.md
```

All 39 rows passed parity/finite checks.

Fastest rows by resolution:

```text
128px:
  gradcache_skip_feature_grad 201.4 ms total / 141.0 ms backward
  fused_first3_sigmoid_mse 221.2 ms total / 159.3 ms backward
  gradcache_reduce_feature_grad_vec4 326.7 ms total / 258.0 ms backward

256px:
  gradcache_skip_feature_grad 822.4 ms total / 519.1 ms backward
  fused_first3_sigmoid_mse 944.3 ms total / 579.0 ms backward
  gradcache_reduce_feature_grad_vec4 1094.9 ms total / 804.7 ms backward
  gradcache 1128.3 ms total / 838.3 ms backward

512px:
  gradcache_skip_feature_grad 1713.5 ms total / 803.7 ms backward
  fused_first3_sigmoid_mse 1724.7 ms total / 833.8 ms backward
  gradcache_cached_bins 1978.6 ms total / 1102.8 ms backward
  gradcache_reduce_feature_grad_cached_bins 1992.8 ms total / 1082.6 ms backward
  linear_sigmoid_mse_skip_colorizer_grad 1995.2 ms total / 949.4 ms backward
```

Cached-bin deltas:

```text
128px:
  direct_atomic -> direct_atomic_cached_bins:
    379.1 -> 347.4 ms total, 304.2 -> 279.1 ms backward
  gradcache -> gradcache_cached_bins:
    350.7 -> 346.0 ms total, 279.4 -> 274.9 ms backward

256px:
  direct_atomic -> direct_atomic_cached_bins:
    1224.4 -> 1151.0 ms total, 892.3 -> 868.9 ms backward
  gradcache -> gradcache_cached_bins:
    1128.3 -> 1384.1 ms total, 838.3 -> 898.5 ms backward

512px:
  direct_atomic -> direct_atomic_cached_bins:
    3795.3 -> 2066.5 ms total, 1837.8 -> 1160.6 ms backward
  gradcache -> gradcache_cached_bins:
    2020.4 -> 1978.6 ms total, 1134.8 -> 1102.8 ms backward
```

## Interpretation

Cached-bin reuse is correct and sometimes helps the isolated renderer, but its
effect is mixed across resolution and it already failed to improve the
first-class 512px trainer row. Keep it as a diagnostic sidecar.

`gradcache_skip_feature_grad` remains the strongest diagnostic row at every
resolution. The feature-gradient atomic path is still a real shader target, but
the trainable scalar/vec4 reductions tried so far are not the right topology for
first-class training. The next shader should be optimized fixedbin or a two-pass
feature-gradient accumulation path, ideally paired with an image-space VJP or
handoff that avoids dense F32 `FeatureToColor` backprop.

## Validation

- `py_compile` passed for `direct_feature_mode_matrix.py`.
- Cached-mode dry run produced the expected command lines.
- Full sequential matrix completed without timeouts or failed rows.
