# Overlap-Key Pressure Diagnostics

Added shared overlap-key telemetry so renderer benchmarks and debug metrics can
measure the real sort workload `K` instead of only reporting splat count `G`.

Implemented `src/train/renderers/overlap_metrics.py` with three query models:

- `custom_rect`: mirrors the current PyTorch tiled renderer's axis-aligned
  rectangular tile assignment.
- `taichi_obb`: mirrors the current Taichi mapper's opacity-aware ellipse bounds
  plus oriented-box tile rejection.
- `exact_conic`: uses the raw-Metal-style exact conic/tile rectangle test.

The main benchmark now prints per-case overlap stats and a summary table:

- total overlap keys
- `K/G`
- p95/max tiles touched per splat
- p95/max splats per tile
- exact-vs-Taichi and exact-vs-custom key ratios when both are enabled

Configs updated:

- `src/benchmark_configs/splat_renderer_benchmark.jsonc`
- `src/benchmark_configs/splat_renderer_64k_throughput.jsonc`
- `src/benchmark_configs/splat_renderer_taichi_metal_scale.jsonc`
- `src/benchmark_configs/splat_renderer_taichi_experiments.jsonc`

Also extended `src/train/debug_metrics.py` so `with_metrics.renderer` logs
`TileDiag/*` overlap pressure for real decoded training outputs. This gives us
the same kind of K/G and large-splat signal on trained scenes, where floaters
matter more than in random splat benchmarks.

Validation:

```bash
uv run python -m py_compile \
  src/train/renderers/overlap_metrics.py \
  src/benchmarks/splat_renderer_benchmark.py \
  src/train/debug_metrics.py
```

passed.

Tiny Taichi probe:

```bash
uv run python src/benchmarks/splat_renderer_benchmark.py \
  --config src/benchmark_configs/splat_renderer_taichi_experiments.jsonc \
  --renderers taichi_reference \
  --resolutions 64 \
  --splat-counts 1024 \
  --sets-per-case 1 \
  --warmup-iters 0 \
  --timed-iters 1 \
  --forward-only \
  --no-save-images \
  --fail-fast
```

Observed:

- `exact_conic`: `K=3358`, `K/G=3.279`
- `taichi_obb`: `K=3660`, `K/G=3.574`
- Taichi OBB generated `1.090x` the exact keys.

64k 128px Taichi probe:

```bash
uv run python src/benchmarks/splat_renderer_benchmark.py \
  --config src/benchmark_configs/splat_renderer_64k_throughput.jsonc \
  --renderers taichi \
  --resolutions 128 \
  --splat-counts 65536 \
  --sets-per-case 1 \
  --warmup-iters 1 \
  --timed-iters 1 \
  --forward-only \
  --no-save-images \
  --fail-fast
```

Observed:

- `exact_conic`: `K=237788`, `K/G=3.628`
- `taichi_obb`: `K=259859`, `K/G=3.965`
- Taichi OBB generated `1.093x` the exact keys.
- `taichi_metal_reference` forward was `25.742ms`.

64k 1024px Taichi probe:

```bash
uv run python src/benchmarks/splat_renderer_benchmark.py \
  --config src/benchmark_configs/splat_renderer_64k_throughput.jsonc \
  --renderers taichi \
  --resolutions 1024 \
  --splat-counts 65536 \
  --sets-per-case 1 \
  --warmup-iters 1 \
  --timed-iters 1 \
  --forward-only \
  --no-save-images \
  --fail-fast
```

Observed:

- `exact_conic`: `K=238251`, `K/G=3.635`
- `taichi_obb`: `K=261777`, `K/G=3.994`
- Taichi OBB generated `1.099x` the exact keys.
- `taichi_metal_reference` forward was `36.389ms`.

Small custom-rect probe:

```bash
uv run python src/benchmarks/splat_renderer_benchmark.py \
  --config src/benchmark_configs/splat_renderer_benchmark.jsonc \
  --renderers custom_tiled \
  --resolutions 64 \
  --splat-counts 512 \
  --sets-per-case 1 \
  --warmup-iters 0 \
  --timed-iters 1 \
  --forward-only \
  --no-save-images \
  --fail-fast
```

Observed:

- `custom_rect`: `K=4757`, `K/G=9.291`
- `exact_conic`: `K=1691`, `K/G=3.303` with 16px tiles
- `exact_conic_custom_tile`: `K=3688`, `K/G=7.203` with custom's 8px tiles
- `taichi_obb`: `K=1819`, `K/G=3.553`
- Custom rectangular bounds generated `1.290x` the exact keys at the same 8px
  tile size. The earlier 2.8x number was a tile-size mismatch between custom's
  8px tiles and Taichi/exact's 16px tiles.

Current interpretation:

- For random splats, Taichi's current OBB mapper is already close to exact
  tile/conic intersection. AccuTile-style exactness would probably save only
  about 8-10% of keys on these synthetic cases.
- The old custom rectangular tiled renderer is sloppier, but the same-tile-size
  gap was ~29% in the checked 64px/512 case, not the misleading 2.8x ratio from
  comparing 8px custom tiles to 16px exact tiles.
- The remaining Taichi speed issue is more likely kernel structure / sort-raster
  fusion than sloppy key generation, unless real trained outputs show floaters
  and a much larger `K/G`.
