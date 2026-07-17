# STAR UVT Cached-Chunks V-JEPA Target Gate

## Goal

Continue the STAR UVT fast feature-shader plan by executing the next concrete
V-JEPA target gate: test whether the 64f/512px target/loss bucket was mostly
repeated target interpolation and whether caching the adapted target layout is a
usable short-run speed path.

## Change

- Added `feature_target.materialization="cached_chunks"` to
  `src/train/train_star_uvt_feature_overfit.py`.
- Added config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_cachedchunks_lr005_5step.jsonc`.
- Added adapter coverage in `tests/test_star_uvt_feature_target_adapter.py`.
- Regenerated:
  - `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_cachedchunks_lr005_5step.json`
  - `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  - `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.json`
  - `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md`
  - `outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.json`
  - `outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.md`
  - `outputs/benchmarks/2026-05-19_star_uvt_target_cache_budget.json`

## Result

The cached-chunks run uses the same source and loss as the streaming chunked
V-JEPA gate:

```text
source: [1,8192,768] V-JEPA tokens
channel-adapted source: [32,32,16,16]
logical target: [64,32,512,512]
materialization: cached_chunks
cached chunks: 32
cached target size: 2048.0 MiB
target load/prep: 2043.8 ms
```

The 5-step cache-hit row passed:

| materialization | step | backward | render | target/loss | loss | overflow |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `chunked` | 3.743s | 1.077s | 0.816s | 1.734s | 1.000014 -> 0.999545 | 0 |
| `cached_chunks` | 1.655s | 0.770s | 0.601s | 0.202s | 1.000014 -> 0.999545 | 0 |

Interpretation:

- The old `1.734s` target/loss bucket was mostly repeated target
  interpolation/materialization, not feature-cache loading.
- `cached_chunks` is the short-run V-JEPA target speed path and is now faster
  than the matched Gaussian/token recon-only cached-conditioning reference
  (`3.460s/step`).
- The tradeoff is explicit: `2048MiB` resident target cache at 64f/512px/F32.
  Before longer runs or higher feature/resolution settings, run a memory-ceiling
  gate or move to target-grid/native-VJP loss.
- The cache budget report makes the immediate scaling cliff concrete:
  128f/512px/F32 is `4GiB`, 64f/512px/F64 is `4GiB`, 64f/1024px/F32 is `8GiB`,
  and 128f/1024px/F64 would be `32GiB` in float32 adapted-target storage.

## Updated Plan

The STAR V-JEPA lane is no longer blocked on simple target caching. Next gates:

1. Benchmark `cached_chunks` memory ceiling across frame count/resolution/F.
2. Prototype target-grid loss or native VJP to avoid resident 2 GiB adapted
   targets.
3. Keep scalar fixedbin/tile-slot feature-gradient accumulation alive for the
   RGB-target feature route.
4. Keep Gaussian/token 512px promotion blocked until NaN guardrails are fixed.

## Validation

Passed:

```bash
rtk .venv/bin/python -m py_compile src/train/train_star_uvt_feature_overfit.py
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py -q
rtk env PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train_star_uvt_feature_overfit.py src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_cachedchunks_lr005_5step.jsonc
rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py
rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py --out-json outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.json --out-md outputs/benchmarks/2026-05-19_star_uvt_precomputed_vjepa_bridge_audit.md
rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/target_cache_budget.py
```
