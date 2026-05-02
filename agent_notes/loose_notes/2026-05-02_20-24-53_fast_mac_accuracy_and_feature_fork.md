# Fast-mac Accuracy Parity And Feature Fork Follow-up

## Context

Follow-up objective after the RGB fast-mac audit:

- verify v6_refined throughput does not trade off accuracy
- answer why v6_refined was not made the default
- create a feature-splatting path that can be selected separately from the
  older `v5_features` baseline

Prior state from `2026-05-02_19-59-25_fast_mac_variant_audit.md`:

- RGB `v6_refined` was wired behind `render.fast_mac.rgb_variant`
- default remained `v5`
- feature splatting remained `v5_features`
- standalone RGB matrix proved v5 high-res backward is much slower than v6/v9
- trainer throughput was favorable for unconditioned TokenGS but mixed for
  free-splats 16-frame runs

## Quality Parity Harness

Added:

- `research_experiments/vjepa_performance/compare_fast_mac_quality.py`

The script runs fixed-seed short training for a set of RGB or feature variants
and writes JSONL with:

- train loss before/after
- elapsed and steps/s
- eval loss/L1/MSE/SSIM/PSNR
- temporal eval metrics when available

### RGB v5 vs v6_refined, 20 steps, seed 7

Unconditioned TokenGS, `128px`, `16f`, `8192` splats:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/compare_fast_mac_quality.py \
  --config src/train_configs/local_mac_compare_unconditioned_tokens_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --variants v5,v6_refined \
  --steps 20 \
  --seed 7 \
  --render-size 128 \
  --clip-length 16 \
  --splat-count 8192 \
  --output-jsonl outputs/benchmarks/fast_mac_quality_unconditioned_tokens_v5_vs_v6_refined_128_16f_20step_seed7_2026-05-02.jsonl
```

| Variant | steps/s | train loss | eval loss | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| `v5` | 2.128 | 0.174283 | 0.180709 | 15.013 | 0.228 |
| `v6_refined` | 2.292 | 0.170204 | 0.179886 | 15.058 | 0.229 |

Free-splats, `128px`, `16f`, `8192` splats:

```bash
PYTHONPATH=src/train WANDB_MODE=disabled uv run python \
  research_experiments/vjepa_performance/compare_fast_mac_quality.py \
  --config src/train_configs/local_mac_compare_free_splats_16f_implicit_camera_128_fast_mac_8192splats.jsonc \
  --variants v5,v6_refined \
  --steps 20 \
  --seed 7 \
  --render-size 128 \
  --clip-length 16 \
  --splat-count 8192 \
  --output-jsonl outputs/benchmarks/fast_mac_quality_free_splats_v5_vs_v6_refined_128_16f_20step_seed7_2026-05-02.jsonl
```

| Variant | steps/s | train loss | eval loss | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| `v5` | 2.350 | 0.346848 | 0.354953 | 8.157 | 0.205 |
| `v6_refined` | 2.705 | 0.346263 | 0.354633 | 8.160 | 0.210 |

Interpretation: the short fixed-seed quality gate did not show accuracy
degradation. v6_refined was slightly better on these two 20-step checks.

This does not by itself justify making v6_refined the default. The earlier
throughput-only matrix showed mixed timings for free-splats 16f, and v6_refined
does not cover feature splatting. Keep default `v5` until the larger train-loop
matrix is repeated with lower variance and the feature story is settled.

## Feature Fork

Added a new fast-mac submodule variant:

- `third_party/fast-mac-gsplat/variants/v6_refined_features/`

Trainer wiring:

- `src/train/renderers/fast_mac.py` now accepts
  `render.fast_mac.feature_variant = "v6_refined_features"`
- default feature path remains `v5_features`

Config:

- `src/train_configs/local_mac_unconditioned_tokens_features_F32_LN_kaiming_g4_v6_refined_features.jsonc`

Important caveat: this is currently a separate namespace fork derived from the
tested `v5_features` F-channel + alpha path. It is not yet a full port of
v6_refined's active-tile/adaptive-stop RGB kernels to arbitrary feature
channels. The reason to keep it is isolation: future F32 speed work can happen
without mutating the stable `v5_features` baseline.

### Feature Accuracy Checks

Command:

```bash
( cd third_party/fast-mac-gsplat/variants/v6_refined_features
  PYTHONPATH=. uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/feature_contract_check.py
  PYTHONPATH=. uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/alpha_output_check.py
  PYTHONPATH=. uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python tests/reference_check.py )
```

Saved output:

- `outputs/benchmarks/fast_mac_v6_refined_features_reference_checks_2026-05-02.txt`

Result:

- shape contract: ok
- F=3 v5 parity: max abs `0`
- F=32 feature grad max abs `5.82e-10`
- alpha tests A-E passed
- reference image max error `5.96e-08`
- saturated conics grad max error `5.96e-08`
- presorted eval/image max error `0`

### F32 Trainer Smoke

Command used a temporary 1-step config from the checked-in v6_refined_features
F32 config, with image/video logging enabled:

```bash
PYTHONPATH=src/train WANDB_MODE=offline uv run python \
  src/train/train_video_token_implicit_dynamic.py /tmp/smoke_f32_v6_refined_features.jsonc
```

Result:

- passed
- offline W&B run: `wandb/offline-run-20260502_202026-rhdvekew`
- exercised alpha-aware feature splatting, colorize, PCA media, and validation
  videos

### F32 Throughput

Trainer F32, `64px`, `8f`, `2048` splats:

| Feature variant | steps/s | frames/s | ms/frame | render median | backward median |
|---|---:|---:|---:|---:|---:|
| `v5_features` | 4.032 | 32.26 | 31.00 | 16.6ms | 116.0ms |
| `v6_refined_features` | 5.034 | 40.27 | 24.83 | 15.1ms | 94.9ms |

Fixed-seed F32 quality, 20 steps, seed 7:

| Feature variant | steps/s | train loss | eval loss | PSNR | SSIM |
|---|---:|---:|---:|---:|---:|
| `v5_features` | 2.748 | 0.315210 | 0.389597 | 7.172 | 0.231 |
| `v6_refined_features` | 3.325 | 0.315235 | 0.389516 | 7.172 | 0.232 |

Standalone projected F32 matrix (`512,2048` x `8192,65536`) says the current
feature fork is not uniformly faster:

| Variant | Res | Splats | Total | Forward | Backward |
|---|---:|---:|---:|---:|---:|
| `v5_features` | 512 | 8192 | 90.616ms | 12.492ms | 78.028ms |
| `v6_refined_features` | 512 | 8192 | 121.737ms | 25.038ms | 99.993ms |
| `v5_features` | 512 | 65536 | 343.272ms | 44.678ms | 295.419ms |
| `v6_refined_features` | 512 | 65536 | 365.594ms | 44.326ms | 320.861ms |
| `v5_features` | 2048 | 8192 | 245.623ms | 23.129ms | 219.511ms |
| `v6_refined_features` | 2048 | 8192 | 303.977ms | 33.974ms | 261.940ms |
| `v5_features` | 2048 | 65536 | 866.810ms | 88.426ms | 773.137ms |
| `v6_refined_features` | 2048 | 65536 | 859.535ms | 87.967ms | 783.782ms |

The trainer microbench improved, likely due to run variance and process state;
the standalone projected matrix is the cleaner raster evidence and says this
namespace fork has not inherited v6_refined's RGB speed advantage.

## Decision

- RGB `v6_refined`: safe for targeted TokenGS runs; not default yet.
- Feature `v6_refined_features`: wired and accuracy-tested; use as an isolated
  experiment namespace, not as a speed win yet.
- Default remains RGB `v5` and feature `v5_features`.

## Next Work

1. Port v6_refined's active-tile/adaptive-stop kernels to arbitrary `F` instead
   of relying on the namespace fork.
2. Repeat throughput with more iterations and fixed process ordering before any
   default promotion.
3. Add a direct quality-parity row for the actual multicam/V-JEPA F32 config if
   we plan to use `v6_refined_features` there.
