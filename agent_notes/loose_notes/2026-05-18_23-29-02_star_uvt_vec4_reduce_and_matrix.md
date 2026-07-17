# STAR UVT Vec4 Reduce And Direct Mode Matrix

Date: 2026-05-18 23:29 Asia/Ho_Chi_Minh

## Goal

Repeat the STAR UVT fast feature-shader plan in docs, fill missed details, and
execute the next bounded shader/benchmark gate. The current open bottleneck was
the F32 feature-gradient path: skip-feature-gradient rows were fast, while the
first trainable scalar reduced-gradient path was slower than gradcache.

## Code Changes

- Added `gradcache_reduce_feature_grad_vec4` to
  `third_party/fast-mac-gsplat/variants/star_uvt_v0/csrc/metal/star_uvt_kernels.metal`.
  It keeps the same direct feature backward semantics as
  `gradcache_reduce_feature_grad`, but packs per-channel feature-gradient
  reduction into `float4` SIMD reductions.
- Exposed the mode through
  `torch_gsplat_bridge_star_uvt.feature_rasterize.direct_atomic_feature_backward`.
- Added trainer-facing `feature_direct_gradcache_reduce_vec4` to
  `src/train/train_star_uvt_feature_overfit.py`.
- Added checked config
  `src/train_configs/star_uvt_feature_testvideo_64f_256_gradcache_reduce_vec4_chunk4_32768t_alpha1_72_cap256_20step.jsonc`.
- Added direct-mode matrix runner
  `research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py`.

## Commands And Results

Build:

```bash
rtk sh -lc '( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/star_uvt_v0 && uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )'
```

Tiny parity:

```bash
rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --backward-mode gradcache_reduce_feature_grad_vec4 \
  --skip-timing \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_tiny_parity.json
```

Result: pass. Max backward error was `9.54e-06` across F4/F32.

Direct synthetic timing:

```bash
rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_kernel_benchmark.py \
  --feature-dims 4,32 --timing-frames 64 --timing-size 256 \
  --timing-tubes 32768 --timing-feature-dim 32 \
  --backward-mode gradcache_reduce_feature_grad_vec4 \
  --timing-warmup 2 --timing-repeat 5 \
  --out-json outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_vec4_serial_64f_256_32768_f32.json
```

Result: pass and finite, zero overflow. `648.0ms` total, `163.3ms` forward,
`484.7ms` backward.

Sequential controls:

```text
gradcache:
  outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_serial_rerun7_after_vec4_64f_256_32768_f32.json
  690.9ms total / 528.2ms backward

scalar reduce:
  outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_reduce_serial_rerun2_after_vec4_sequential_64f_256_32768_f32.json
  721.8ms total / 516.4ms backward

skip feature grad:
  outputs/benchmarks/2026-05-18_star_uvt_feature_direct_metal_gradcache_skip_feature_grad_serial_rerun3_after_vec4_sequential_64f_256_32768_f32.json
  553.6ms total / 326.5ms backward
```

Important caveat: an earlier scalar-reduce and skip-feature control were run in
parallel and clearly fought for MPS. Those JSONs were renamed to
`.invalid_parallel.txt` and should not be used for report rows.

First-class cap256 real-video trainer:

```bash
rtk env STAR_UVT_TILE_CAPACITY=256 PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 WANDB_MODE=offline .venv/bin/python \
  src/train/train.py \
  src/train_configs/star_uvt_feature_testvideo_64f_256_gradcache_reduce_vec4_chunk4_32768t_alpha1_72_cap256_20step.jsonc
```

Result:

```text
pass: true
loss: 0.31889 -> 0.29290
zero overflow, max tile 252/256
mean step: 2094.8ms
mean forward: 388.8ms
mean color/loss: 198.2ms
mean backward: 1412.5ms
```

Same-session first-class controls:

```text
feature_direct_gradcache:
  1807.0ms/step, 1333.1ms backward

feature_direct_gradcache_reduce:
  1889.7ms/step, 1394.8ms backward
```

Read: vec4 is not a first-class cap256 win. It remains selectable as a
diagnostic, but `feature_direct_gradcache` stays the current fastest valid
feature mode.

Sequential direct-mode matrix:

```bash
rtk env PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 .venv/bin/python \
  research_experiments/star_uvt_feature_tubes/direct_feature_mode_matrix.py \
  --sizes 128,256 --frames 64 --tubes 32768 --feature-dim 32 \
  --feature-dims 4,32 --warmup 1 --repeat 3 \
  --out-dir outputs/benchmarks/2026-05-18_star_uvt_feature_direct_mode_matrix_128_256_64f_32768t_f32 \
  --timeout-sec 180
```

Key matrix rows:

```text
128px:
  gradcache: 332.7ms total / 285.1ms backward / overflow 8093
  vec4 reduce: 315.5ms total / 256.2ms backward / overflow 8093
  logit handoff: 303.4ms total / 237.4ms backward / 15.4ms prep / overflow 8093

256px:
  gradcache: 962.5ms total / 731.3ms backward / zero overflow
  scalar reduce: 989.0ms total / 756.9ms backward / zero overflow
  vec4 reduce: 968.2ms total / 718.8ms backward / zero overflow
  fused first3: 696.1ms total / 465.5ms backward / zero overflow
  logit handoff: 1256.2ms total / 661.3ms backward / 121.8ms prep / zero overflow
```

The matrix is a lower-repeat sequential table, not a replacement for the
repeat-5 artifacts. Its main value is reproducibility and preventing accidental
parallel-MPS timing contamination.

## Interpretation

Vec4 proved the scalar channel loop was part of the synthetic cost, but not the
real cap256 trainer bottleneck. The first-class slowdown suggests the remaining
problem is the per-contributor barrier/topology and cap256 tile pressure, not
just scalar SIMD math.

Next useful shader work should be one of:

- true optimized fixedbin feature backward that exploits zero-overflow support
  without the current direct-path topology
- two-pass/sidecar feature-gradient accumulation that changes the feature
  atomic/reduction topology
- image-space RGB-grad handoff only if it avoids both full in-tile colorizer
  reduction and per-pixel `W^T` over all F channels

Do not promote linear handoff, logit handoff, scalar reduce, or vec4 reduce as
the default trainer path. Use `feature_direct_gradcache` for current valid
feature training and `feature_direct_fixedbin` as the promotion/fallback guard.

## Docs Updated

- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `EXPERIMENTS.md`
- `PROJECT_INDEX.md`
- `TODO/README.md`
- `README.md`
- `agent_notes/key_learnings.md`
- `outputs/benchmarks/2026-05-18_renderer_scaling_report.md`
- `outputs/benchmarks/2026-05-18_star_uvt_feature_firstclass_scale_report.md`
