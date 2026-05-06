# Feature Kernel Fork Iteration

## Context

The goal was to keep stable baselines untouched while iterating on forked Metal
feature-splatting kernels for the high-pressure `B=16,G=8192,F=32,512px` case.
The stable `v6_refined_features` variant remained clean; all new shader work
went into new forks.

## Work Done

Forks created:

- `variants/v6_refined_features_f32_reduce`
  - previous atomic-reduction fork
  - now has the safe benchmark runner
  - now avoids allocating a full `g_colors` gradient tensor when color gradients
    are skipped
- `variants/v6_refined_features_f32_stage`
  - stages `[GSP_CHUNK, GSP_FEATURE_CAP]` feature/color data in threadgroup
    memory
  - has a fail-fast estimated threadgroup-memory guard
- `variants/v6_refined_features_f32_accum`
  - accumulates direct fast-forward feature output in per-thread local storage
    for `F <= 32`, then writes the dense output once
  - direct fast eval/state only; active/overflow paths remain inherited
  - now has opt-in trainer dispatch; synthetic wins are shape-dependent
- `variants/v6_refined_features_f32_gradcache`
  - caches each pixel thread's `grad_features[pix, :]` vector for direct fast
    backward when `F <= 32`
  - now has opt-in trainer dispatch after passing parity and fixed-render timing
- `variants/v6_refined_features_f32_block4`
  - reuses generic feature-gradient loads in 4-channel blocks
  - passed correctness gates but regressed timing, so it remains benchmark-only
- `variants/v6_refined_features_f32_fixedbin`
  - removes the exact-length bin allocation path for no-overflow fast-path rows
  - allocates a fixed `tile_count * max_fast_pairs` int32 ID buffer and raises
    on overflow instead of falling back
  - passed feature/alpha and trainer fixed-render parity gates; remains opt-in

Shared doc added:

- `research_notes/fast_mac_feature_metal_performance.md`

Saved benchmark-runner artifacts:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_safe_matrix_dry_run.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_safe_matrix_smoke.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_safe_matrix_warm_smoke.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_safe_batch_strategy_matrix.jsonl`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_current_warm4.json`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_f32_reduce_warm4.json`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_current_breakdown1.json`
- `benchmark_outputs/trainer_phase/unconditioned_f32_v6_refined_features_f32_reduce_breakdown1.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_baseline_vs_reduce.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_freeze_colors_matrix_dry_run.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_default_cap2048_dry_run.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_default_cap2048_smoke.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_reduce_trainable_rerun.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_reduce_freeze_colors.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f3_feature_baseline_vs_reduce.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f3_feature_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_feature_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_baseline_vs_reduce_cap2048.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_reduce_cap2048_freeze_colors.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_accum_cap2048_stdout.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_accum_cap2048_stdout.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_b16_g8192_f32_accum_cap2048_freeze_colors_stdout.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_b16_g8192_f32_cap_profile.jsonl`
- `benchmark_outputs/trainer_phase/multicam256_f32_v5_features_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v5_features_warm1_iters2_cachehit.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_breakdown_warm1.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_breakdown_warm1.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_f32_reduce_fixed_render_smoke.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v5_features_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_freeze_colors_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_freeze_colors_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_reduce_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_accum_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_reduce_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_accum_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_v5_features_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_f32_reduce_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_v5_features_breakdown_warm1.json`
- `benchmark_outputs/trainer_phase/multicam128_f32_f32_reduce_breakdown_warm1.json`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_block4_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_block4_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_gradcache_confirm_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_f32_b16_fixedbin_smoke_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_fixedbin_matrix.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_fixedbin_matrix.jsonl`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_gradcache_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_gradcache_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json`
- `benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_heldout_v6_vs_f32_fixedbin_fixed_render_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_fixedbin_fixed_render_grad_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_fixedbin_fixed_render_seed0_warm2_iters4.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json`

## Validation

Both new forks passed:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/<fork>/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/<fork>/tests/alpha_output_check.py
```

Observed for both forks:

```text
shape contract active_policy=off: ok
F=3 v5 parity active_policy=off max_abs=0
shape contract active_policy=on: ok
F=3 v5 parity active_policy=on max_abs=0
F=3 feature grad active_policy=off max_abs=1.8626451e-09
F=8 feature grad active_policy=off max_abs=9.3132257e-10
F=32 feature grad active_policy=off max_abs=2.3283064e-10
F=32 feature grad active_policy=on max_abs=2.3283064e-10
F=32 no-NaN smoke active_policy=off: ok
F=32 no-NaN smoke active_policy=on: ok
Test A passed.
Test B passed.
Test C passed.
Test D passed.
Test E passed.
Test F passed.
```

The later `f32_fixedbin` fork also passed the same feature/alpha gates, including
the F64 gradient parity row (`F=64 feature grad active_policy=off
max_abs=1.1641532e-10`). It then passed exact fixed-render trainer parity
against stable `v6_refined_features`: 256px train/heldout forward parity had
zero feature/alpha/RGB/loss diff, and 128px train/heldout gradient parity
matched decoded sequence grads within `1.14e-09`.

## Benchmarks

Local MPS, `512x512`, `B=16`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, `batch_strategy=flatten`, `active_policy=off`.

| Variant | iters | forward ms | backward ms | total mean ms | total median ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| stable `v6_refined_features` | 3 | 216.6 | 1712.1 | 1928.7 | 1919.7 |
| `v6_refined_features_f32_reduce` | 5 | 180.2 | 535.3 | 715.4 | 704.4 |
| `v6_refined_features_f32_stage` | 3 | 183.2 | 755.8 | 938.9 | 941.9 |
| `v6_refined_features_f32_accum` | 5 | 168.9 | 610.1 | 779.1 | 749.5 |

Post no-color-gradient allocation cleanup, same shape:

| Variant | freeze colors | iters | forward ms | backward ms | total mean ms | total median ms |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| `v6_refined_features_f32_reduce` | false | 5 | 123.0 | 446.0 | 568.9 | 515.9 |
| `v6_refined_features_f32_reduce` | true | 5 | 119.7 | 242.5 | 362.2 | 348.7 |

Forward-only eval:

| Variant | iters | forward mean ms | forward median ms |
| --- | ---: | ---: | ---: |
| `v6_refined_features_f32_reduce` | 5 | 202.6 | 202.6 |
| `v6_refined_features_f32_stage` | 5 | 276.7 | 271.1 |
| `v6_refined_features_f32_accum` | 5 | 168.0 | 165.4 |

Small safe batch-strategy matrix, `128x128`, `G=1024`, `F=32`, `iters=2`:

| B | Strategy | Total mean ms | Forward ms | Backward ms |
| ---: | --- | ---: | ---: | ---: |
| 1 | flatten | 42.4 | 29.4 | 13.0 |
| 1 | serial | 44.5 | 29.6 | 15.0 |
| 4 | flatten | 39.0 | 17.2 | 21.7 |
| 4 | serial | 83.1 | 52.8 | 30.3 |
| 16 | flatten | 109.5 | 35.7 | 73.8 |
| 16 | serial | 350.2 | 228.9 | 121.3 |

Tiny trainer phase trace, checked-in 64px F32 config:

| Variant | warmup/iters | Total mean ms | Raster fwd ms | Autograd backward total ms |
| --- | --- | ---: | ---: | ---: |
| stable `v6_refined_features` | 2/4 | 363.0 | 17.3 | 172.5 |
| `v6_refined_features_f32_reduce` | 2/4 | 292.3 | 13.2 | 150.1 |

Detached backward probes at 64px were not consistent enough to promote:

| Variant | raster backward probe ms |
| --- | ---: |
| stable `v6_refined_features` | 13.2 |
| `v6_refined_features_f32_reduce` | 20.9 |

Larger synthetic check, `128x128`, `B=16`, `G=8192`, `F=32`,
`case=medium_sigma_3_8`, `warmup=1`, `iters=3`:

| Variant | Total mean ms | Forward ms | Backward ms |
| --- | ---: | ---: | ---: |
| stable `v6_refined_features` | 974.9 | 664.8 | 310.2 |
| `v6_refined_features_f32_reduce` | 931.2 | 656.0 | 275.2 |

128px multicam trainer phase, checked-in DeepView train cameras
`0006/0014`, heldout `0005`, `16` frames, `8192` splats, cached V-JEPA
features, `warmup=1`, `iters=2`:

| Variant | Total mean ms | Raster fwd ms | Autograd backward total ms |
| --- | ---: | ---: | ---: |
| current config, `v5_features` | 439.9 | 25.8 | 245.3 |
| opt-in `v6_refined_features_f32_reduce` | 403.5 | 22.0 | 241.4 |

Warmed one-iteration detached backward probes on the same 128px multicam shape:

| Variant | Raster backward probe ms | Loss/colorize probe ms | Project probe ms | Model probe ms |
| --- | ---: | ---: | ---: | ---: |
| current config, `v5_features` | 74.0 | 100.9 | 61.6 | 96.7 |
| opt-in `v6_refined_features_f32_reduce` | 81.5 | 101.1 | 59.0 | 50.9 |

Frozen-feature synthetic rerun on the 128px synthetic shape:

| Variant | Colors trainable | Total mean ms | Forward ms | Backward ms |
| --- | --- | ---: | ---: | ---: |
| `v6_refined_features_f32_reduce` | true | 1055.7 | 794.6 | 261.1 |
| `v6_refined_features_f32_reduce` | false | 1002.7 | 852.5 | 150.2 |

The forward timings drifted upward in both adjacent reruns, so they should not
be compared against the earlier `931.2ms` row as a semantic change. The useful
signal is backward: skipping color gradients still removes a large chunk of
work at `B=16,F=32`.

Runtime-cap correction:

The first safe matrix runner default forced `GSP_CHUNK=32` and
`GSP_FAST_CAP=512`. That made F3 feature paths look catastrophically slower
than RGB. Re-running with the trainer-like/runtime default `64/2048` changed
the diagnosis:

| Shape | Variant | Total mean ms | Forward ms | Backward ms |
| --- | --- | ---: | ---: | ---: |
| 128px B16/G8192/F3 | RGB `v6_refined` | 27.6 | 12.1 | 15.5 |
| 128px B16/G8192/F3 | stable `v6_refined_features` | 40.9 | 12.2 | 28.8 |
| 128px B16/G8192/F3 | `v6_refined_features_f32_reduce` | 28.7 | 11.8 | 16.9 |
| 128px B16/G8192/F32 | stable `v6_refined_features` | 196.9 | 25.6 | 171.3 |
| 128px B16/G8192/F32 | `v6_refined_features_f32_reduce` | 93.5 | 26.1 | 67.4 |
| 128px B16/G8192/F32 | `v6_refined_features_f32_accum` | 91.2 | 24.3 | 66.9 |
| 512px B16/G8192/F32 | stable `v6_refined_features` | 1001.9 | 88.2 | 913.7 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce` | 449.2 | 123.0 | 326.2 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum` | 388.1 | 77.8 | 310.3 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_reduce`, colors frozen | 267.7 | 87.9 | 179.8 |
| 512px B16/G8192/F32 | `v6_refined_features_f32_accum`, colors frozen | 256.7 | 78.4 | 178.3 |

This is now the most important benchmark lesson from the session: do not use
`cap=512` for primary throughput claims. It stress-tests fallback behavior and
can invert the feature/RGB story. With `64/2048`, F3 feature forward is fine;
stable F3 backward is the remaining generic-gradient overhead; the reduction
fork recovers RGB-class F3 backward and gives the big F32 backward win. Under
the corrected cap, the accumulation fork is also back in play for training
timing because it beats `f32_reduce` on the synthetic fwd+bwd pressure rows.

Profile confirmation on the 128px/B16/G8192/F32 forward-only shape:

| Chunk | Cap | Forward ms | Overflow tiles | Max pairs/tile | Mean pairs/tile |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 512 | 833.9 | 970 | 1135 | 875.7 |
| 32 | 2048 | 23.8 | 0 | 1135 | 875.7 |
| 64 | 512 | 833.1 | 970 | 1135 | 875.7 |
| 64 | 2048 | 23.5 | 0 | 1135 | 875.7 |

So the slow path was cap overflow, not chunk size and not inherent F-channel
forward overhead.

256px trainer phase follow-up:

| Variant | warmup/iters | Total mean ms | Total median ms | Raster fwd ms | Autograd backward total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `v5_features` current config | 1/2 cache hit | 844.3 | 844.3 | 69.5 | 589.0 |
| stable `v6_refined_features` | 1/2 | 842.6 | 842.6 | 71.6 | 588.4 |
| `v6_refined_features_f32_reduce` | 1/2 | 883.1 | 883.1 | 78.8 | 562.5 |
| stable `v6_refined_features` | 2/4 | 959.1 | 980.3 | 73.6 | 661.6 |
| `v6_refined_features_f32_reduce` | 2/4 | 1188.4 | 1147.8 | 89.5 | 656.0 |

Warmed 1-iteration backward probes at 256px:

| Variant | Raster backward probe ms | Loss/colorize probe ms | Project probe ms | Model probe ms |
| --- | ---: | ---: | ---: | ---: |
| stable `v6_refined_features` | 151.0 | 595.5 | 95.3 | 294.9 |
| `v6_refined_features_f32_reduce` | 116.6 | 391.6 | 63.3 | 76.1 |

Interpretation: the fork still improves the isolated raster-backward probe, but
the whole trainer step is noisy and not a clear default-promotion win. Encode
and model phases vary enough between runs that the current phase benchmark is
good for warnings but not a final renderer-selection gate.

Fixed-render benchmark mode was added to `src/benchmarks/trainer_phase_benchmark.py`.
It samples/decodes one clip, clones detached Gaussian leaves each measured
iteration, and times only project/raster/loss/backward. It also accepts
`--seed` so model init, sampling, and random background are comparable across
variant runs.

Seeded 256px fixed-render graph, `seed=0`, `warmup=2`, `iters=4`:

| Variant | Colors trainable | Total mean ms | Total median ms | Raster fwd ms | Autograd backward total ms |
| --- | --- | ---: | ---: | ---: | ---: |
| `v5_features` current config | true | 672.1 | 672.3 | 65.8 | 526.5 |
| stable `v6_refined_features` | true | 674.4 | 673.3 | 66.7 | 527.9 |
| `v6_refined_features_f32_reduce` | true | 659.5 | 658.9 | 67.7 | 511.5 |
| `v6_refined_features_f32_accum` | true | 665.2 | 665.5 | 73.7 | 511.8 |
| `v6_refined_features_f32_reduce` | false | 608.0 | 608.9 | 68.7 | 457.7 |
| `v6_refined_features_f32_accum` | false | 611.4 | 609.4 | 76.9 | 453.3 |

Later same-session rerun after adding `f32_gradcache` trainer dispatch:

| Variant | Total mean ms | Total median ms | Raster fwd ms | Autograd backward total ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 1133.8 | 1122.8 | 89.7 | 888.6 | `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json` |
| `v6_refined_features_f32_reduce` | 1165.6 | 1113.1 | 87.5 | 923.5 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json` |
| `v6_refined_features_f32_accum` | 1060.0 | 1073.4 | 93.8 | 807.6 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_gradcache.json` |
| `v6_refined_features_f32_gradcache` | 1008.0 | 966.4 | 82.6 | 765.6 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4.json` |

This is a cleaner timing gate than full-step trainer timing, but the local MPS
absolute timings drifted between windows. The first fixed-render window favored
`f32_reduce` slightly; the later same-window comparison favored `f32_gradcache`.
The frozen-color path is materially faster and remains relevant for camera-only
or frozen-splat follow-up runs. Still do not make any fork the default without
heldout-quality parity.

Fixed-render parity script added:

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train WANDB_MODE=disabled WANDB_SILENT=true \
  .venv/bin/python src/benchmarks/fixed_render_variant_parity.py \
  /tmp/dynaworld_phase_multicam_256_v6_refined_features.jsonc \
  /tmp/dynaworld_phase_multicam_256_f32_reduce.jsonc \
  --seed 0 --json-output benchmark_outputs/trainer_phase/multicam256_v6_vs_f32_reduce_fixed_render_parity_seed0.json
```

Results:

| Baseline | Candidate | Max feature diff | Max alpha diff | Max RGB diff | Loss diff |
| --- | --- | ---: | ---: | ---: | ---: |
| stable `v6_refined_features` | `f32_reduce` | 0.0 | 0.0 | 0.0 | 0.0 |
| stable `v6_refined_features` | `f32_accum` | 0.0 | 0.0 | 0.0 | 0.0 |
| stable `v6_refined_features` | `f32_gradcache` | 0.0 | 0.0 | 0.0 | 0.0 |
| stable `v6_refined_features`, heldout | `f32_reduce` | 0.0 | 0.0 | 0.0 | 0.0 |
| stable `v6_refined_features`, heldout | `f32_accum` | 0.0 | 0.0 | 0.0 | 0.0 |
| stable `v6_refined_features`, heldout | `f32_gradcache` | 0.0 | 0.0 | 0.0 | 0.0 |

This is not heldout-quality parity, but it is a cheap pre-flight gate: on the
seeded train and heldout trainer render paths, both forks produce exactly the
same features, alpha, RGB, and reconstruction loss as stable.

I then extended the same script with `--check-gradients` and ran the cheaper
128px multicam graph against both train and heldout targets. The forward tensors,
loss, and colorize-MLP parameter gradients were exact; decoded sequence gradients
matched within MPS reduction noise:

| Baseline | Candidate | Target | Max sequence grad diff | Artifact |
| --- | --- | --- | ---: | --- |
| stable `v6_refined_features` | `f32_reduce` | train | 8.15e-10 | `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_reduce_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `f32_reduce` | heldout | 8.15e-10 | `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_reduce_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `f32_accum` | train | 4.07e-10 | `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_accum_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `f32_accum` | heldout | 9.60e-10 | `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_accum_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `f32_gradcache` | train | 8.15e-10 | `benchmark_outputs/trainer_phase/multicam128_train_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json` |
| stable `v6_refined_features` | `f32_gradcache` | heldout | 7.86e-10 | `benchmark_outputs/trainer_phase/multicam128_heldout_v6_vs_f32_gradcache_fixed_render_grad_parity_seed0.json` |

This is still not a W&B heldout-quality run, but it is the right cheap
correctness gate before spending that time: the forked backward paths are
training-gradient equivalent to the stable feature kernel on the seeded trainer
render/loss graph.

After the matrix runner was extended to support multiple fork variants and
`GSP_FEATURE_CAP`, I ran conservative F64 pressure rows. I also added F64 to the
fork-local `feature_contract_check.py` color-gradient parity loop; both
`f32_reduce` and `f32_accum` passed with `F=64 feature grad max_abs=1.16e-10`.

| Shape | Stable total | `f32_reduce` total | `f32_accum` total | Read | Artifact |
| --- | ---: | ---: | ---: | --- | --- |
| 128px B16/G8192/F64 | 678.4ms | 340.6ms | 365.6ms | reduce wins | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_f64_b8_b16_variant_matrix.jsonl` |
| 256px B16/G8192/F64 | 2021.6ms | 1224.9ms | 964.6ms | accum wins | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_variant_matrix.jsonl` |
| 256px B16/G8192/F64 frozen colors | n/a | 645.9ms | 628.0ms | both much faster, accum slight win | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_freeze_colors_variant_matrix.jsonl` |
| 512px B4/G8192/F64 | 754.0ms | 412.7ms | 620.3ms | reduce wins | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b4_variant_matrix.jsonl` |
| 512px B8/G8192/F64 | 1614.0ms | 890.8ms | 837.6ms | accum slight win | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b8_variant_matrix.jsonl` |

Important nuance: `f32_accum` currently sets `GSP_LOCAL_ACCUM_CAP=32`, so true
F64 local output accumulation is not active yet. The F64 wins are real measured
variant timings but probably come from compiler/layout side effects plus shared
backward changes, not from the intended local-accum mechanism. A clean next fork
should test `GSP_LOCAL_ACCUM_CAP=64` or a two-block 32-channel accumulator.

I then made that clean next fork:

- `variants/v6_refined_features_f64_accum64`
- copied from `f32_accum`, separate package/op namespace
- default `GSP_LOCAL_ACCUM_CAP=64`
- stable `v6_refined_features`, `f32_reduce`, and `f32_accum` untouched

Build and correctness gates passed:

```bash
( cd third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64/tests/feature_contract_check.py
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_refined_features_f64_accum64/tests/alpha_output_check.py
```

The result is a useful negative. The cap-64 fork passed F64 gradient parity, but
it did not beat the existing forks:

| Shape | `f32_reduce` | `f32_accum` | `f64_accum64` | Read | Artifact |
| --- | ---: | ---: | ---: | --- | --- |
| 256px B16/G8192/F64 warm2/iters3 | 654.5ms | 646.4ms | 682.4ms | cap64 forward cost loses | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_accum64_confirm_matrix.jsonl` |
| 512px B8/G8192/F64 warm1/iters2 | 542.7ms | 567.9ms | 567.0ms | no win; reduce best | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b8_accum64_matrix.jsonl` |

I also filled in the high-batch boundaries without jumping to 4K:

| Shape | Stable | `f32_reduce` | `f32_accum` | Read | Artifact |
| --- | ---: | ---: | ---: | --- | --- |
| 256px B32/G8192/F32 | 2117.0ms | 1002.9ms | 1049.7ms | reduce slight win at B32 | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_variant_matrix.jsonl` |
| 512px B16/G8192/F64 forward/profile | 613.9ms | 734.4ms | 706.2ms | no overflow; direct forward stable fastest | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b16_profile_forward.jsonl` |
| 512px B16/G8192/F64 frozen colors | n/a | 1952.3ms | 1811.5ms | lower-memory backward viable | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b16_freeze_colors_variant_matrix.jsonl` |
| 512px B16/G8192/F64 trainable forks | n/a | 2280.8ms | 2429.1ms | actual pressure boundary; reduce wins | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f64_b16_trainable_forks.jsonl` |

I tried one more fork from the audit:

- `variants/v6_refined_features_f32_gradcache`
- copied from `f32_reduce`
- direct fast backward only
- for non-F3 `feature_dim <= 32`, cache `grad_features[pix, :]` in
  `thread float grad_cache[32]`
- stable and existing forks untouched

It built and passed `feature_contract_check.py` plus `alpha_output_check.py`.
The result is noisy but worth keeping as an opt-in trainer candidate:

| Shape | `f32_reduce` | `f32_accum` | `f32_gradcache` | Read | Artifact |
| --- | ---: | ---: | ---: | --- | --- |
| 256px B16/G8192/F32 | 337.1ms | 314.3ms | 273.7ms | grad cache wins moderate B | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 256px B32/G8192/F32 | 585.0ms | 593.8ms | 609.2ms | private cache loses at B32 | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_gradcache_matrix.jsonl` |
| 512px B16/G8192/F32 early | 527.0ms | 596.4ms | 597.2ms | early target row loses | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_gradcache_matrix.jsonl` |
| 512px B16/G8192/F32 confirm | 423.3ms | 412.9ms | 391.3ms | later confirm wins | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_gradcache_confirm_matrix.jsonl` |

I then tried the smaller-cache idea as its own fork:

- `variants/v6_refined_features_f32_block4`
- copied from `f32_reduce`
- computes `dot(grad_features, color)` and `g_colors` reduction from the same
  channel loads in 4-channel blocks
- stable and existing forks untouched

It built and passed the fork-local correctness gates, but timing regressed:

| Shape | `f32_reduce` | `f32_accum` | `f32_gradcache` | `f32_block4` | Read | Artifact |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 256px B16/G8192/F32 | 332.8ms | 326.0ms | 326.4ms | 377.9ms | block4 loses | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_block4_matrix.jsonl` |
| 256px B32/G8192/F32 | 656.7ms | 647.2ms | 620.8ms | 692.8ms | block4 loses | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_block4_matrix.jsonl` |
| 512px B16/G8192/F32 | 437.6ms | 393.1ms | 369.7ms | 547.3ms | block4 loses badly | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_block4_matrix.jsonl` |

The fixed-bin host/kernel fork tested a different hypothesis: exact-size
`binned_ids` allocation and `final_offset.item()` sync might matter at larger
batch/resolution. This fork is no-overflow only and costs a fixed ID buffer
instead of a dynamically sized one.

| Shape | `f32_reduce` | `f32_accum` | `f32_gradcache` | `f32_fixedbin` | Read | Artifact |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 128px B16/G8192/F32 | 213.2ms | n/a | 197.1ms | 176.7ms | fixedbin wins smoke | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_128_f32_b16_fixedbin_smoke_matrix.jsonl` |
| 256px B16/G8192/F32 | 493.2ms | 520.0ms | 497.3ms | 523.4ms | fixedbin loses | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 256px B32/G8192/F32 | 1003.4ms | 1048.0ms | 968.9ms | 996.6ms | fixedbin beats reduce, loses gradcache | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f32_b16_b32_fixedbin_matrix.jsonl` |
| 512px B16/G8192/F32 | 855.4ms | 706.5ms | 716.4ms | 501.8ms | fixedbin wins target row | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_512_f32_b16_fixedbin_matrix.jsonl` |
| 256px B16/G8192/F64 | 779.8ms | 726.0ms | n/a | 718.1ms | fixedbin slight win | `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_256_f64_b16_fixedbin_matrix.jsonl` |

Latest 256px fixed-render trainer window after adding `f32_fixedbin`:

| Variant | Total mean ms | Total median ms | Raster fwd ms | Autograd backward total ms | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 725.4 | 718.8 | 69.4 | 572.2 | `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_accum` | 696.9 | 691.5 | 76.3 | 537.2 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_gradcache` | 814.7 | 828.0 | 79.3 | 638.4 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_seed0_warm2_iters4_rerun_after_fixedbin.json` |
| `v6_refined_features_f32_fixedbin` | 696.1 | 687.7 | 68.8 | 544.4 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_fixedbin_fixed_render_seed0_warm2_iters4.json` |

Sampled-memory trainer rerun, same 256px fixed-render setup,
`seed=0`, `warmup=1`, `iters=2`, `--memory-sample-interval-ms 1.0`:

| Variant | Total mean ms | Raster fwd ms | Autograd backward total ms | Sampled peak current bytes | Artifact |
| --- | ---: | ---: | ---: | ---: | --- |
| stable `v6_refined_features` | 673.9 | 68.0 | 525.8 | 1412745984 | `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_fixedbin` | 649.8 | 68.5 | 500.6 | 1700988160 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_fixedbin_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_accum` | 670.2 | 75.5 | 513.0 | 1412745984 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_reduce` | 667.4 | 75.8 | 510.5 | 1636237568 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_sampled_memory_seed0_warm1_iters2.json` |
| `v6_refined_features_f32_gradcache` | 649.3 | 67.0 | 502.3 | 1412745984 | `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_sampled_memory_seed0_warm1_iters2.json` |

## Interpretation

- No fork should replace the stable baseline yet. Keep `v6_refined_features`
  untouched and choose forked variants explicitly in experiment configs.
- Atomic reduction remains the simplest proven train-path fork. `f32_gradcache`
  won one same-session 256px fixed-render window, while `f32_fixedbin` and
  `f32_accum` tied in the later same-session window. All need heldout-quality
  parity before default promotion.
- The no-color-gradient allocation cleanup is useful for frozen-feature or
  camera-only follow-up runs; the `--freeze-colors` B16/F32/512 probe measured
  `362.2ms` total / `242.5ms` backward after the cleanup.
- Threadgroup feature staging is not worth promoting; it increases backward
  pressure enough to lose total time.
- Local feature accumulation is useful for eval/forward-only, but its trainer
  wins are shape/load-dependent. Under the corrected `64/2048` cap, it can win
  synthetic F32 train-path rows, but the fixed-render trainer gate has not made
  it an obvious default.
- The `f32_reduce` trainer dispatch works behind
  `render.fast_mac.feature_variant = "v6_refined_features_f32_reduce"`, but the
  64px phase trace is not a promotion gate because raster is only a small part
  of the step and detached raster-backward probes were inconsistent.
- The 128px multicam trainer trace is a better smoke and is directionally
  faster with the fork, but it still is not a promotion gate by itself. The
  detached raster-backward probe was slower for the fork, and non-raster
  backward phases remain large enough to hide raw kernel improvements.
- The earlier F=3 feature-path overhead was mostly a runtime-cap artifact,
  not an inherent generic-feature forward cost. With `64/2048`, F3 feature
  forward matches RGB; stable feature backward remains slower because it keeps
  generic per-channel gradient handling, and the reduction fork fixes most of
  that F3 backward gap.
- The safe matrix runner now supports `--freeze-colors` for the fork and
  explicitly skips stable-baseline rows in that mode. That keeps the known-good
  stable benchmark script untouched while still measuring camera-only or
  frozen-splat follow-up paths.
- The safe matrix runner now defaults to `GSP_CHUNK=64,GSP_FAST_CAP=2048` to
  match trainer-like throughput. Use `cap=512` only as an explicit
  fallback-pressure stress case.
- The 256px trainer phase check does not justify making `f32_reduce` the
  default. It lowers the detached raster-backward probe, but full-step medians
  remain mixed or slower; future trainer benchmarks need fixed-sample or
  fixed-decoded-graph isolation before we trust full-step deltas.
- The fixed-render graph benchmark resolves part of that noise, but same-session
  comparisons are still necessary. One window favored `f32_reduce`; a later
  same-session window favored `f32_gradcache`. It is timing evidence, not
  quality evidence.
- The fixed-render parity script gives a cheap correctness gate before long
  train runs. At 256px it showed exact output/loss parity for `f32_reduce`,
  `f32_accum`, `f32_gradcache`, and `f32_fixedbin` versus stable v6 features,
  including the heldout camera.
- The `f32_block4` fork is a useful negative: sharing per-channel loads between
  dot-product and color-gradient reduction looked reasonable but increased
  backward enough to lose every tested row.
- The `f32_fixedbin` fork is a useful but bounded host/binning result. Removing
  the exact-size bin allocation sync wins the target synthetic
  `512px/B16/F32` row and ties `f32_accum` on the latest 256px fixed-render
  trainer graph, but sampled trainer memory is worse than stable/gradcache and
  it loses `256px/B16/F32`. It costs a fixed ID buffer of about `128 MiB` at
  `512px/B16/tile16/cap2048`. Keep it no-overflow and opt-in until a
  heldout-quality run validates it over optimizer time.
- The current 256px timing/memory compromise points at `f32_gradcache`, not
  fixedbin: it tied fixedbin for total time while matching the stable sampled
  current allocation. Treat this as a bounded row, not promotion proof.
- The next non-kernel lever is trainer microbatch/framewise backward, because
  dense `[B,H,W,F]` surfaces are still structural batch pressure.
- Next kernel forks worth trying: a lower-private-pressure `float4` block cache
  rather than a full F32 grad vector; a two-block F64 accumulator only if we
  revisit F64 local accumulation; and active/overflow local accumulation only
  if profiles show those paths matter. The fixed-cap fast-bin host/kernel path
  has now been tried as `f32_fixedbin`.

## Safety Notes

- No heavy 4K/64K or overflow-stress sweeps were run.
- High-pressure probes stayed at already-measured local-safe tensor sizes:
  `512px,B=16,G=8192,F=32`, `512px,B=8,G=8192,F=64`, and
  `256px,B=32,G=8192,F=32`, with at most 5 measured iterations and sequential
  launches.
- The safe runner defaults to `128px,G=1024,B=1,F=32`, excludes
  `overflow_stress`, supports `--dry-run`, and uses per-job timeouts.
