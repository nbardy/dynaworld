# v9 Feature Shader Benchmark

Date: 2026-05-08 16:55

## Context

The user asked for a "best of all worlds" feature-splatting shader pass:
combine the known good `gradcache` and `zero_bg` ideas, compare against the
older v7/v8/v9 lineages, and benchmark before promoting anything.

This note records the concrete artifacts and results.

## New Fork

New opt-in variant:

```text
third_party/fast-mac-gsplat/variants/v9_features_gradcache_zero_bg/
```

Lineage:

- copied from `v6_refined_features_f32_gradcache`
- ported the zero-feature-background final tail-skip from
  `v6_refined_features_f32_zero_bg`
- unique package/op namespace:
  `torch_gsplat_bridge_v9_features_gradcache_zero_bg`

Root trainer dispatch was wired in:

```text
render.fast_mac.feature_variant = "v9_features_gradcache_zero_bg"
```

No checked-in trainer config was changed to use it by default.

## Alpha/Background Safety Read

The feature-background concern was rechecked before promoting the zero-bg
fork. The old fix is still present: feature splats are colorized first, then
alpha-composited against train-time random RGB in objective space. Therefore a
zero feature background in the rasterizer does not by itself reopen the old
MLP-background-bias shortcut.

Detailed note:

```text
agent_notes/loose_notes/2026-05-08_16-25-46_alpha_bg_bleed_features.md
```

Remaining risk is low-alpha edge bleed, not fully transparent pixels training
the colorizer.

## v7/v8/v9 Lineage Audit

Read-only audit result:

- `v7*`: RGB/hardware/front-K experiments, not good F32 feature-port bases.
- `v8_hw_*` and most `v9_hw_*`: hardware interop/eval probes, not trainable
  F32 feature paths.
- `v8_project3d`/`v9_project3d_train`: relevant only if 3D-to-2D projection
  shows up as the measured bottleneck; not a feature-rasterizer fix.
- the only near-term idea worth a future fork is v8-style host metadata parsing
  to avoid hot-path `meta_f32.cpu()` parsing in the F32 feature variants.

Decision: skip v7/v9 hardware feature ports for now. Treat host-metadata as a
possible future `v10`, but do not mix it into the v9 combined fork.

## Direct Benchmark Matrix

Shape:

```text
B=16, G=8192, F=32, case=medium_sigma_3_8
GSP_TILE_SIZE=16, GSP_CHUNK=64, GSP_FAST_CAP=2048, GSP_FEATURE_CAP=64
backward=true, active_policy=off, warmup=3, iters=8
```

Artifacts:

```text
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_feature_variants_B16_G8192_F32_256_active_off.jsonl
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_feature_variants_B16_G8192_F32_512_active_off.jsonl
```

| Variant | 256 mean / fwd / bwd ms | 512 mean / fwd / bwd ms | Read |
| --- | ---: | ---: | --- |
| `v6_refined_features` | 771.669 / 77.172 / 694.497 | 1005.936 / 91.949 / 913.987 | stable baseline |
| `v6_refined_features_f32_reduce` | 311.416 / 76.223 / 235.193 | 403.377 / 92.223 / 311.154 | large win vs stable |
| `v6_refined_features_f32_gradcache` | 291.201 / 75.870 / 215.331 | 367.351 / 88.994 / 278.357 | best existing candidate |
| `v6_refined_features_f32_zero_bg` | 311.211 / 74.613 / 236.597 | 400.173 / 89.354 / 310.819 | reduce + tail skip, no gradcache |
| `v9_features_gradcache_zero_bg` | 291.516 / 76.602 / 214.914 | **364.049 / 86.369 / 277.679** | parity-safe combined candidate |

Read: v9 is basically gradcache plus a small 512px forward/total win. It does
not beat gradcache at 256 in this matrix, and it does not combine into a new
large speed jump.

## Active Tile Check

Artifacts:

```text
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_v6_refined_features_f32_gradcache_B16_G8192_F32_{256,512}_active_{off,on}.jsonl
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_v9_features_gradcache_zero_bg_B16_G8192_F32_{256,512}_active_{off,on}.jsonl
```

| Variant | Shape | Active off mean ms | Active on mean ms | Read |
| --- | --- | ---: | ---: | --- |
| `f32_gradcache` | 256 | 274.225 | 334.917 | active on slower |
| `f32_gradcache` | 512 | 372.667 | 526.546 | active on much slower |
| `v9` | 256 | 274.334 | 334.884 | same ranking |
| `v9` | 512 | 368.283 | 527.979 | active off remains best |

Read: for this dense B16/G8192 row, force active tiles off.

## Trainer Fixed-Render Parity

Artifact:

```text
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_fixed_render_parity_v6_refined_features_vs_v9_features_gradcache_zero_bg_256.json
```

Gate:

```text
src/benchmarks/fixed_render_variant_parity.py
baseline:  v6_refined_features
candidate: v9_features_gradcache_zero_bg
target: train
check_gradients: true
```

Result:

```text
loss_abs_diff: 0.0
max_feature_abs_diff: 0.0
max_alpha_abs_diff: 0.0
max_rgb_abs_diff: 0.0
max_colorize_grad_abs_diff: 0.0
max_sequence_grad_abs_diff: 8.149072527885437e-10
```

This covers the real trainer fixed-render graph, including alpha-aware
composition and colorizer gradients.

## Decision

`v9_features_gradcache_zero_bg` is a valid opt-in candidate. It is safe enough
for a short trainer smoke or a trainer-profile A/B, but not yet enough to
replace stable configs by default.

Recommended next config experiment:

```text
feature_variant = "v9_features_gradcache_zero_bg"
alpha_threshold = 1/128
active_policy = "off"
use_active_tiles = false
feature_background = 0.0
losses.background.train_mode = "random_rgb"
```

Do not promote v7/v8/v9 hardware lineages into F32 training yet. If another
kernel fork is worth trying, make it a narrow host-metadata fork from the F32
feature line, not a hardware/RGB port.

## Follow-Up: v10 and v11

The literal v10/v11 part of the prompt was completed after the first v9 pass.
Both forks are opt-in and keep stable baselines untouched:

```text
third_party/fast-mac-gsplat/variants/v10_features_gradcache_zero_bg_hostmeta/
third_party/fast-mac-gsplat/variants/v11_features_gradcache_zero_bg_hostmeta_fixedbin/
```

Lineage:

- `v10_features_gradcache_zero_bg_hostmeta`: copied from v9 and ports the v8
  host-side metadata split. `meta_i32/meta_f32` still go to Metal kernels, but
  the bridge parses CPU `meta_host_i32/meta_host_f32` for allocation,
  validation, and dispatch sizing instead of calling `.cpu()` on MPS metadata.
- `v11_features_gradcache_zero_bg_hostmeta_fixedbin`: copied from v10 and ports
  the fixed-capacity no-overflow binning idea from
  `v6_refined_features_f32_fixedbin`. It allocates
  `tile_count * max_fast_pairs` IDs, initializes offsets on GPU, and raises if
  any tile exceeds the cap.

Both variants build and pass:

```text
python -m py_compile .../rasterize.py .../benchmark_mps.py .../benchmark_matrix.py
uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .../tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python .../tests/alpha_output_check.py
```

The F32 feature contract checks reported exact F=3 parity, F=32 gradient max
diff `2.3283064e-10`, and alpha-output tests A-F passed for both v10 and v11.

## v9/v10/v11 Direct Matrix

Same-session target matrix:

```text
B=16, G=8192, F=32, case=medium_sigma_3_8
GSP_TILE_SIZE=16, GSP_CHUNK=64, GSP_FAST_CAP=2048, GSP_FEATURE_CAP=64
backward=true, active_policy=off, warmup=3, iters=8
```

Artifacts:

```text
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_v9_v10_v11_256_B16_G8192_F32.jsonl
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_v9_v10_v11_512_B16_G8192_F32.jsonl
```

| Variant | 256 mean / fwd / bwd ms | 512 mean / fwd / bwd ms | Read |
| --- | ---: | ---: | --- |
| `v9_features_gradcache_zero_bg` | 273.189 / 70.754 / 202.435 | 367.900 / 89.877 / 278.023 | combined gradcache+zero-bg |
| `v10_features_gradcache_zero_bg_hostmeta` | 271.220 / 69.237 / 201.982 | 366.133 / 88.372 / 277.761 | small hostmeta win |
| `v11_features_gradcache_zero_bg_hostmeta_fixedbin` | **270.471 / 68.298 / 202.173** | **364.049 / 86.256 / 277.793** | best direct row, no-overflow only |

Read: host metadata is real but small. Fixedbin+hostmeta gives the best direct
row, mostly by reducing forward/bin overhead. It does not materially change the
F32 backward cost.

## v10/v11 Trainer Parity and Timing

Trainer fixed-render parity versus v9, 256px train target with gradients:

```text
v10:
loss_abs_diff: 0.0
max_feature_abs_diff: 0.0
max_alpha_abs_diff: 0.0
max_rgb_abs_diff: 0.0
max_colorize_grad_abs_diff: 0.0
max_sequence_grad_abs_diff: 1.0186340659856796e-09

v11:
loss_abs_diff: 0.0
max_feature_abs_diff: 0.0
max_alpha_abs_diff: 0.0
max_rgb_abs_diff: 0.0
max_colorize_grad_abs_diff: 0.0
max_sequence_grad_abs_diff: 6.111804395914078e-10
```

Artifacts:

```text
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_fixed_render_parity_v9_vs_v10_features_gradcache_zero_bg_hostmeta_256_train.json
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_fixed_render_parity_v9_vs_v11_features_gradcache_zero_bg_hostmeta_fixedbin_256_train.json
```

Short fixed-render trainer timing, 256px, seed 0, warmup 1, iters 3:

| Variant | total median ms | raster fwd median ms | backward median ms | Read |
| --- | ---: | ---: | ---: | --- |
| `v9_features_gradcache_zero_bg` | 911.913 | 74.493 | 744.056 | baseline for this pass |
| `v10_features_gradcache_zero_bg_hostmeta` | 942.831 | 76.453 | 764.017 | worse in noisy trainer pass |
| `v11_features_gradcache_zero_bg_hostmeta_fixedbin` | **897.462** | **69.505** | **731.843** | best short trainer timing |

Artifacts:

```text
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_trainer_fixed_render_v9_features_gradcache_zero_bg_256_seed0_warm1_iters3.json
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_trainer_fixed_render_v10_features_gradcache_zero_bg_hostmeta_256_seed0_warm1_iters3.json
benchmark_outputs/fast_mac_feature_kernels/2026-05-08_trainer_fixed_render_v11_features_gradcache_zero_bg_hostmeta_fixedbin_256_seed0_warm1_iters3.json
```

Decision: `v11_features_gradcache_zero_bg_hostmeta_fixedbin` is the best opt-in
candidate from this loop. It is not a universal default because it is
no-overflow only and fixedbin increases ID-buffer memory, but it is the most
reasonable "best of all worlds" feature shader to try in a short trainer run.
