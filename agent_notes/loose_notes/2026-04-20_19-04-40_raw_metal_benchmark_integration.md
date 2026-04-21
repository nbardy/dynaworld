# Raw Metal Benchmark Integration

Integrated `shader_experiments/raw_metal_fast_rasterizer/` into the existing
splat speed and accuracy benchmark surface.

Changes:

- Added `src/benchmarks/raw_metal_mlx_bridge.py` as the shared optional MLX
  bridge.
- Added `raw_metal` renderer key to `src/benchmarks/splat_renderer_benchmark.py`
  for forward-only throughput runs.
- Generalized `src/benchmarks/splat_renderer_accuracy.py` from Taichi-only to
  multi-renderer accuracy specs and added `raw_metal`.
- Added `raw_metal` sections to the accuracy, 64k throughput, and Taichi
  experiment JSONC configs.
- Made speed and accuracy renderer construction lazy by requested key so
  raw-metal-only runs do not initialize Taichi or other optional backends.

Raw artifact fixes discovered by smoke testing:

- `mx.fast.metal_kernel` names were built from the config cache key and included
  invalid Metal identifier characters like `=`, `.`, and `|`. Added a sanitized
  kernel symbol builder.
- `mx.eval(total_pairs_arr).item()` failed because `mx.eval()` returns `None` in
  the tested MLX path. Changed it to evaluate first, then read `.item()`.

Validation run:

- `uv run python -m py_compile src/benchmarks/raw_metal_mlx_bridge.py src/benchmarks/splat_renderer_benchmark.py src/benchmarks/splat_renderer_accuracy.py shader_experiments/raw_metal_fast_rasterizer/mlx_projected_gaussian_rasterizer.py`
- `uv run --with mlx python src/benchmarks/splat_renderer_accuracy.py --config src/benchmark_configs/splat_renderer_accuracy.jsonc --renderers raw_metal --no-save-images --fail-on-mismatch`
  - Passed 18/18 cases across 16/32px and 4/8/16 splats.
  - Worst observed raw Metal image max error was below `1e-7`; packed/features
    gradient errors stayed around `1e-9`.
- `uv run --with mlx python src/benchmarks/splat_renderer_accuracy.py --renderers taichi,raw_metal --resolutions 16 --splat-counts 4 --sets-per-case 1 --no-save-images --fail-on-mismatch`
  - Taichi and raw Metal both passed against the direct CPU float64 Torch packed
    reference.
- `uv run --with mlx python src/benchmarks/splat_renderer_benchmark.py --renderers raw_metal --resolutions 64 --splat-counts 1024 --sets-per-case 1 --warmup-iters 1 --timed-iters 3 --forward-only --no-save-images`
  - Raw Metal forward-only smoke: about `8.3 ms` mean forward for 64x64/G=1024
    in this local run, including Torch/NumPy/MLX bridge overhead.
- `uv run python src/benchmarks/splat_renderer_benchmark.py --renderers raw_metal --resolutions 8 --splat-counts 1 --sets-per-case 1 --warmup-iters 0 --timed-iters 1 --forward-only --no-save-images`
  - Without MLX in the project env, raw Metal skipped cleanly and printed the
    `uv run --with mlx ...` install hint.
- `uv run --with mlx python src/benchmarks/splat_renderer_benchmark.py --renderers raw_metal --resolutions 4096 --splat-counts 65536 --sets-per-case 1 --warmup-iters 0 --timed-iters 1 --forward-only --no-save-images`
  - First-use 4K/64K forward-only smoke completed, but included setup/compile
    effects: about `940.8 ms` forward.
- `uv run --with mlx python src/benchmarks/splat_renderer_benchmark.py --renderers raw_metal --resolutions 4096 --splat-counts 65536 --sets-per-case 1 --warmup-iters 1 --timed-iters 3 --forward-only --no-save-images`
  - Warmed 4K/64K forward-only smoke completed: about `86.1 ms` mean forward
    (`11.6 FPS`) in this local run, including the benchmark's Torch/NumPy/MLX
    bridge overhead and output conversion.

Important interpretation note: the raw Metal/MLX rasterizer uses the same
`x + 0.5` pixel-center convention as the Taichi-style packed-2D accuracy
baseline. The existing Dynaworld training renderers use integer pixel grid
coordinates, so throughput `compare_outputs` against `custom_*` or
`vectorized_sparse` can show expected diffs until those conventions are aligned.

MLX was absent from the project env. Use `uv run --with mlx ...` for raw Metal
benchmark runs unless MLX is added to the project dependencies. The bridge
reports a clean skip with an install hint when MLX is unavailable.
