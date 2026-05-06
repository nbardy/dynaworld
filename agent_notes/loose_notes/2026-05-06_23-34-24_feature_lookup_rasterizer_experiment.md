# Feature Lookup Rasterizer Experiment

Created an isolated prototype fork:

```text
third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/
```

The starting point was `v6_refined_features`, because that fork already has the
F-channel Metal path, accumulated alpha, active-tile scheduling, and backward
contract needed to evaluate whether the F32 path is memory-bound.

Feasibility read:

- A compact-basis path is feasible without changing compositing math: splat K
  compact coefficients with the existing dense-channel kernel, then reconstruct
  full F features with `compact @ lookup`.
- This should reduce rasterizer output/intermediate/backward-channel pressure
  when K is much smaller than F. It still materializes the final `[H,W,F]`
  tensor after lookup, so the trainer path must measure where peak memory lands.
- A true sparse ID/weight Metal path is deeper than a small fork. The existing
  kernel assumes dense `colors[BG,F]` in forward and dense `g_colors[BG,F]` in
  backward. Sparse IDs require new op signatures, new Metal loops over
  `feature_ids[BG,L]`, and an explicit lookup-gradient contract.

Prototype status:

- New namespace: `torch_gsplat_bridge_v6_feature_lookup_experiment`.
- New custom-op namespace: `torch.ops.gsplat_metal_v6_feature_lookup_experiment`.
- Added `rasterize_projected_gaussians_feature_lookup(...)`, which renders K
  compact channels with zero compact background and reconstructs F features with
  an output-space background tail.
- Added `rasterize_projected_gaussians_feature_ids(...)`, but it currently
  densifies IDs to `[G,K]` / `[B,G,K]` before rasterization. This is an API
  skeleton, not the true sparse-kernel memory win.

Validation update:

The fork now builds and has a tiny MPS parity check:

```bash
( cd /Users/nicholasbardy/git/gsplats_browser/dynaworld/third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment
  uv run --project /Users/nicholasbardy/git/gsplats_browser/dynaworld python setup.py build_ext --inplace )
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/tests/feature_lookup_parity_check.py
```

Observed:

```text
features max_abs=8.9406967e-08
alpha max_abs=0
loss max_abs=0
grad_means max_abs=5.8207661e-11
grad_conics max_abs=1.4901161e-08
grad_weights max_abs=6.0535967e-09
grad_lookup max_abs=2.5611371e-09
grad_opacities max_abs=1.8626451e-09
feature lookup direct parity: ok
id_skeleton feature max_abs=0
id_skeleton alpha max_abs=0
feature id skeleton parity: ok
```

Interpretation:

- The compact-basis math is correct for the low-rank case where direct full
  features equal `feature_weights @ lookup`.
- Gradients through the compact rasterizer plus post-raster lookup match the
  direct full-feature rasterizer with a Torch matmul in front.
- This is still not the true sparse-ID memory win, because the ID helper
  densifies to `[G,K]` before rasterization and the final `[H,W,F]` feature
  image still exists after reconstruction.

Bounded timing/memory probe:

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py \
  --height 128 --width 128 --batch-size 4 --gaussians 2048 \
  --feature-dim 32 --compact-dims 4,8,16 --warmup 1 --iters 3 \
  --seed 11 --no-overflow-fallback \
  --jsonl-output benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_128_b4_g2048_f32_k4_8_16.jsonl
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py \
  --height 128 --width 128 --batch-size 16 --gaussians 2048 \
  --feature-dim 32 --compact-dims 4,8,16 --warmup 1 --iters 2 \
  --seed 12 --no-overflow-fallback \
  --jsonl-output benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_128_b16_g2048_f32_k4_8_16.jsonl
```

| Shape | K | Direct mean ms | Lookup mean ms | Read |
| --- | ---: | ---: | ---: | --- |
| 128px B4/G2048/F32 | 4 | 49.9 | 18.9 | lookup faster |
| 128px B4/G2048/F32 | 8 | 42.6 | 22.6 | lookup faster |
| 128px B4/G2048/F32 | 16 | 41.7 | 30.8 | lookup faster |
| 128px B16/G2048/F32 | 4 | 162.2 | 84.0 | lookup faster |
| 128px B16/G2048/F32 | 8 | 164.6 | 100.3 | lookup faster |
| 128px B16/G2048/F32 | 16 | 112.6 | 75.4 | lookup faster |

Memory read:

- The benchmark records sampled `torch.mps.current_allocated_memory()` and
  `driver_allocated_memory()` after synchronized backward, not true peak memory.
- The sampled current allocation is mixed: lookup is similar/slightly higher
  for K=4/8 and lower for the B16/K16 row.
- The final dense `[B,H,W,F]` reconstruction remains live, so this prototype is
  a timing candidate first and not yet proof of the sparse-ID memory thesis.

Sampled-peak update:

The benchmark now also starts a background memory sampler during the measured
forward/backward window. This is not as authoritative as Xcode/Metal counters,
but it is better than only reading memory after synchronized backward.

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py \
  --height 128 --width 128 --batch-size 16 --gaussians 2048 \
  --feature-dim 32 --compact-dims 4,8,16 --warmup 1 --iters 2 \
  --seed 13 --no-overflow-fallback --memory-sample-interval-ms 0.25 \
  --jsonl-output benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_sampled_peak_128_b16_g2048_f32_k4_8_16.jsonl
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python \
  third_party/fast-mac-gsplat/variants/v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py \
  --height 128 --width 128 --batch-size 16 --gaussians 8192 \
  --feature-dim 32 --compact-dims 4,8,16 --warmup 1 --iters 2 \
  --seed 14 --no-overflow-fallback --memory-sample-interval-ms 0.25 \
  --jsonl-output benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_sampled_peak_128_b16_g8192_f32_k4_8_16.jsonl
```

| Shape | K | Direct mean ms | Lookup mean ms | Direct sampled peak | Lookup sampled peak |
| --- | ---: | ---: | ---: | ---: | ---: |
| 128px B16/G2048/F32 | 4 | 158.8 | 81.0 | 152014080 | 117284096 |
| 128px B16/G2048/F32 | 8 | 159.8 | 96.5 | 154112256 | 124100864 |
| 128px B16/G2048/F32 | 16 | 112.6 | 75.4 | 151361024 | 167093504 |
| 128px B16/G8192/F32 | 4 | 225.9 | 96.9 | 244678400 | 166563584 |
| 128px B16/G8192/F32 | 8 | 179.6 | 117.9 | 278758400 | 210605056 |
| 128px B16/G8192/F32 | 16 | 184.1 | 155.6 | 282427392 | 198021120 |

Interpretation:

- Lookup stays faster on the larger `G=8192` pressure row.
- At `G=8192`, sampled current allocation is lower for K=4/8/16.
- At `G=2048`, K=16 is faster but has higher sampled current allocation than
  direct, so the memory win is shape-dependent.
- The driver allocation is still dominated by allocator/page behavior and should
  not be used as the promotion metric.

Next decisive test:

Use a real peak-memory profiler or trainer fixed-render harness before doing
trainer integration. If K only improves synthetic timing but not peak memory,
keep this as a compact-basis speed branch rather than claiming it solves F32
feature memory.
