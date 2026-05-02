# Fast-mac Variant Audit And Trainer Swap

## Context

The thread was about whether our current fast-mac RGB trainer path was still on
an older slow backward. The user remembered a pre-PowerFoam rasterizer where
backward was roughly `2x` forward and around `30ms` for `~64k` splats at
`2k-4k` resolution.

Current trainer dispatch before this pass:

- RGB `F == 3`: `third_party/fast-mac-gsplat/variants/v5`
- feature splatting `F != 3`: `third_party/fast-mac-gsplat/variants/v5_features`

The important distinction is that full trainer steps include data sampling,
frame resize, model decode, projection, rasterization, pixel losses, backward
through everything, and optimizer. The standalone variant matrix isolates the
projected-input rasterizer much better than the full-step timing.

## Build And Reference Checks

Initial reference status:

- already built/passing: `v5`, `v5_features`, `v5_features alpha`, `v6`,
  `v6_upgrade`
- missing compiled ops before build: `v6_refined`, `v8`, `v8_hw_eval`,
  `v8_hw_train`, `v8_project3d`, `v9_project3d_train`
- `v9_hw_tile_exact_probe` was unavailable until its v8 backend was built

Build command shape used from the dynaworld root:

```bash
for v in v6_refined v8 v8_hw_eval v8_hw_train v8_project3d v9_project3d_train v9_hw_tile_exact_probe; do
  ( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/$v
    uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
done
```

After build, all target reference checks passed. Saved logs:

- `outputs/benchmarks/fast_mac_variant_reference_checks_2026-05-02.txt`
- `outputs/benchmarks/fast_mac_variant_reference_checks_after_build_2026-05-02.txt`

## Standalone Variant Matrix

Harness added:

- `research_experiments/vjepa_performance/benchmark_fast_mac_variants.py`

Command:

```bash
PYTHONPATH=src/train uv run python research_experiments/vjepa_performance/benchmark_fast_mac_variants.py \
  --resolutions 512,2048,4096 \
  --gaussians 8192,65536 \
  --warmup 2 \
  --iters 5 \
  --timeout 420 \
  --output-jsonl outputs/benchmarks/fast_mac_variant_matrix_rgb_f3_single_frame_512_2k_4k_8192_65536_2026-05-02.jsonl
```

All 66 rows completed with `status=ok`. Times below are milliseconds. The
regular `benchmark_mps.py` rows report median total plus mean forward/backward
phase values; the v9 full-backward probe reports comparable median phases.

### 2048px / 65536 Splats

| Variant | Total | Forward | Backward | Bwd/Fwd |
|---|---:|---:|---:|---:|
| `v9_hw_tile_exact_probe` | 33.19 | 7.35 | 25.86 | 3.52 |
| `v8` | 34.23 | 8.11 | 27.10 | 3.34 |
| `v6_refined` | 35.85 | 9.69 | 26.18 | 2.70 |
| `v6_upgrade` | 35.86 | 9.94 | 26.08 | 2.63 |
| `v8_hw_train` | 37.25 | 8.73 | 28.01 | 3.21 |
| `v6` | 37.51 | 9.84 | 27.78 | 2.82 |
| `v8_hw_eval` | 38.44 | 10.86 | 30.69 | 2.83 |
| `v9_project3d_train` | 45.17 | 11.85 | 33.76 | 2.85 |
| `v5` | 47.05 | 14.19 | 39.40 | 2.78 |
| `v5_features` | 74.51 | 11.71 | 62.00 | 5.30 |
| `v8_project3d` | 89.06 | 29.77 | 60.99 | 2.05 |

### 4096px / 65536 Splats

| Variant | Total | Forward | Backward | Bwd/Fwd |
|---|---:|---:|---:|---:|
| `v6` | 59.27 | 11.62 | 47.85 | 4.12 |
| `v9_hw_tile_exact_probe` | 59.46 | 10.79 | 48.66 | 4.51 |
| `v6_upgrade` | 60.96 | 12.47 | 49.06 | 3.93 |
| `v6_refined` | 61.55 | 12.79 | 49.01 | 3.83 |
| `v8_hw_eval` | 63.50 | 10.84 | 51.73 | 4.77 |
| `v9_project3d_train` | 66.25 | 12.24 | 54.11 | 4.42 |
| `v8_project3d` | 69.90 | 12.48 | 57.40 | 4.60 |
| `v5_features` | 106.63 | 15.04 | 92.87 | 6.18 |
| `v8` | 146.44 | 30.02 | 94.43 | 3.15 |
| `v8_hw_train` | 149.22 | 42.61 | 108.40 | 2.54 |
| `v5` | 160.67 | 43.55 | 112.25 | 2.58 |

### Interpretation

The user's memory is partly confirmed. The `~30ms` backward class at `64k/2k`
exists in the v6/v8/v9 family: `25.9-30.7ms` for the best rows. It is not true
of the current trainer's v5 path (`39.4ms` at 2k/64k and `112.3ms` at 4k/64k).

At 4k/64k, v5 is a real outlier:

- v5: `160.7ms` total, `112.3ms` backward
- v6/v6_upgrade/v6_refined/v9_hw_tile: `59-62ms` total, `47.8-49.1ms` backward

So yes, v5 backward is blowing up relative to the newer trainable paths. The
best current 4k/64k backward in this matrix is still around `48ms`, not `30ms`.

`v9_hw_tile_exact_probe` is very fast but is a probe wrapper over a v8 compute
replay backend, not a direct trainer integration target. `v6_refined` was chosen
for trainer wiring because it is the corrected v6 handoff, passes reference
checks, and exposes the same projected RGB API the trainer already uses.

## Trainer Integration

Code changes:

- `src/train/renderers/fast_mac.py`
  - added `fast_mac.rgb_variant`, default `v5`
  - added `fast_mac.feature_variant`, default `v5_features`
  - added `rgb_variant="v6_refined"` dispatch for RGB `F == 3`
  - kept feature `F != 3` explicitly on `v5_features`
- `research_experiments/vjepa_performance/benchmark_free_splats_throughput.py`
  - now records `fast_mac_rgb_variant`, `fast_mac_feature_variant`, and
    `fast_mac_batch_strategy`
- new configs:
  - `src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_v6_refined_8192splats.jsonc`
  - `src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_v6_refined_8192splats.jsonc`

Verification:

```bash
PYTHONPATH=src/train uv run python -m py_compile \
  src/train/renderers/fast_mac.py \
  research_experiments/vjepa_performance/benchmark_free_splats_throughput.py \
  research_experiments/vjepa_performance/benchmark_fast_mac_variants.py
```

Tiny trainer smoke passed:

- `outputs/benchmarks/fast_mac_v6_refined_trainer_smoke_free_splats_128_1f_8192_2026-05-02.jsonl`
- canonical 1-step trainer smoke also passed with
  `/tmp/smoke_fast_mac_v6_refined_f3.jsonc`, `WANDB_MODE=offline`, and run dir
  `wandb/offline-run-20260502_200219-03f4s71w`

F32 was not re-wired: `v6_refined` is RGB-only in this trainer path, and the
feature-splatting branch remains `v5_features`.

## Trainer Throughput Reruns

Artifacts:

- `outputs/benchmarks/fast_mac_v6_refined_trainer_free_splats_single_frame_128_512_2k_4k_8192_2026-05-02.jsonl`
- `outputs/benchmarks/fast_mac_v5_trainer_free_splats_single_frame_128_512_2k_4k_8192_2026-05-02.jsonl`
- `outputs/benchmarks/fast_mac_v6_refined_trainer_free_splats_128_512_16f_8192_2026-05-02.jsonl`
- `outputs/benchmarks/fast_mac_v5_trainer_free_splats_128_512_16f_8192_2026-05-02.jsonl`
- `outputs/benchmarks/fast_mac_v6_refined_trainer_unconditioned_tokens_128_512_1f_16f_8192_2026-05-02.jsonl`
- `outputs/benchmarks/fast_mac_v5_trainer_unconditioned_tokens_128_512_1f_16f_8192_2026-05-02.jsonl`

### Free Splats

| Size | Frames | Variant | Steps/s | Frames/s | ms/frame | Render median | Backward median |
|---:|---:|---|---:|---:|---:|---:|---:|
| 128 | 1 | `v6_refined` | 13.45 | 13.45 | 74.33 | 22.3ms | 27.4ms |
| 128 | 1 | `v5` | 3.74 | 3.74 | 267.44 | 35.0ms | 93.8ms |
| 512 | 1 | `v6_refined` | 5.39 | 5.39 | 185.47 | 49.1ms | 81.8ms |
| 512 | 1 | `v5` | 5.71 | 5.71 | 175.05 | 35.8ms | 77.6ms |
| 2048 | 1 | `v6_refined` | 1.44 | 1.44 | 696.42 | 71.0ms | 373.8ms |
| 2048 | 1 | `v5` | 1.09 | 1.09 | 914.27 | 147.6ms | 389.2ms |
| 4096 | 1 | `v6_refined` | 0.065 | 0.065 | 15323.54 | 472.1ms | 2587.2ms |
| 4096 | 1 | `v5` | 0.057 | 0.057 | 17401.24 | 856.4ms | 2193.3ms |
| 128 | 16 | `v6_refined` | 1.44 | 23.03 | 43.42 | 55.4ms | 305.4ms |
| 128 | 16 | `v5` | 1.85 | 29.53 | 33.86 | 59.9ms | 197.0ms |
| 512 | 16 | `v6_refined` | 0.97 | 15.45 | 64.72 | 56.8ms | 523.3ms |
| 512 | 16 | `v5` | 1.28 | 20.42 | 48.97 | 46.0ms | 391.2ms |

Free-splats result is mixed. `v6_refined` improves single-frame 128px and 2k,
and lowers 4k render median, but the 16-frame free-splats cases regress. Do not
make it the global default from these numbers.

### Unconditioned TokenGS

| Size | Frames | Variant | Steps/s | Frames/s | ms/frame | Render median | Backward median |
|---:|---:|---|---:|---:|---:|---:|---:|
| 128 | 1 | `v6_refined` | 15.59 | 15.59 | 64.13 | 12.6ms | 27.9ms |
| 128 | 1 | `v5` | 9.20 | 9.20 | 108.64 | 17.2ms | 45.5ms |
| 128 | 16 | `v6_refined` | 1.53 | 24.52 | 40.77 | 45.5ms | 349.5ms |
| 128 | 16 | `v5` | 0.87 | 13.98 | 71.53 | 53.6ms | 602.0ms |
| 512 | 1 | `v6_refined` | 9.22 | 9.22 | 108.44 | 24.7ms | 44.9ms |
| 512 | 1 | `v5` | 5.12 | 5.12 | 195.46 | 35.2ms | 67.8ms |
| 512 | 16 | `v6_refined` | 0.65 | 10.43 | 95.90 | 45.6ms | 788.8ms |
| 512 | 16 | `v5` | 0.68 | 10.86 | 92.06 | 48.4ms | 882.9ms |

TokenGS result is more favorable. `v6_refined` is `1.69-1.80x` faster in the
128/1f, 128/16f, and 512/1f rows, and roughly parity at 512/16f.

## Current Decision

Keep default trainer behavior on v5 for now, but use
`fast_mac.rgb_variant="v6_refined"` for targeted TokenGS speed runs. It is not
yet a universal replacement because free-splats 16-frame regressed.

The next serious speed work should not only chase the rasterizer:

- high-res 4k full trainer steps are dominated by image resize/loss/backward
  and other full-frame work once the rasterizer gets faster
- `v9_hw_tile_exact_probe` deserves a direct-integration investigation only if
  we want to beat v6_refined further; it is currently a probe wrapper, not a
  clean trainer backend
- for feature splatting, low precision or faster F-channel raster remains a
  separate fork; `v6_refined` does not solve `F=32`

## Follow-up Tests

1. Repeat the TokenGS rows with more iterations and one fixed process order to
   reduce MPS variance.
2. Audit why `v6_refined` regresses free-splats 16-frame backward while helping
   TokenGS. Candidate branches: batch scheduler behavior, free parameter graph
   shape, retained graph chunking, or MPS variance.
3. If promoting `v6_refined`, run a short quality parity smoke:
   same seed, same config, v5 vs v6_refined, compare loss/PSNR/SSIM after a
   small fixed step count.
4. Consider a direct `v9_hw_tile_exact` or `v9_project3d_train` trainer
   integration only after the wrapper/API story is clean.
