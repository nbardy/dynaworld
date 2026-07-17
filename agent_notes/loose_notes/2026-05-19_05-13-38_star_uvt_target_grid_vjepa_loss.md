# STAR UVT Target-Grid V-JEPA Loss Follow-Up

Date: 2026-05-19 05:13 +0700

## Goal

Close the next STAR UVT cached-feature target gate after `cached_chunks`: avoid
the resident multi-GiB adapted target cache while keeping the real V-JEPA target
route measurable at 64f/512px/8192t/F32.

## What Changed

- Added `feature_target.materialization="target_grid"` to
  `src/train/train_star_uvt_feature_overfit.py`.
- The loader now keeps the channel-adapted V-JEPA source grid
  `[32,32,16,16]` resident instead of materializing or caching the dense
  `[64,32,512,512]` render-grid target.
- During the feature-target loss, rendered feature chunks are downsampled to the
  corresponding target-grid slice before MSE.
- Added the test coverage in
  `tests/test_star_uvt_feature_target_adapter.py` for target-grid chunk mapping,
  render-to-target-grid adaptation, and gradient flow through the rendered
  tensor.
- Added the checked config
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_5step.jsonc`.

## Benchmark

Command:

```bash
PYTHONPATH=src/train WANDB_MODE=offline .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_lr005_5step.jsonc
```

Result:

- output:
  `outputs/benchmarks/2026-05-19_star_uvt_feature_vjepa_target_64f_512px_8192t_f32_chunk2_gradcache_reduce_vec4_no_prenorm_targetgrid_lr005_5step.json`
- pass: `true`
- loss: `0.999935 -> 0.999467`
- zero tile overflow
- target grid: `[32,32,16,16]`, `1.0MiB`
- feature target load/prep: `138.35ms`
- mean step: `1350.84ms`
- mean render forward: `547.73ms`
- mean target/loss: `41.02ms`
- mean backward: `705.15ms`
- last step: `1181.61ms`, `587.45ms` backward
- required model gradient flow present for raw features, UV/T centers,
  velocities, precision, and opacity.

## Comparison Read

The regenerated comparison report is
`outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`.

Matched 64f/512px/8192t rows:

- STAR V-JEPA streaming target: `3.743s/step`, `1.077s` backward,
  `1.734s` target/loss.
- STAR V-JEPA cached chunks: `1.655s/step`, `0.770s` backward,
  `0.202s` target/loss, `2048MiB` resident adapted target.
- STAR V-JEPA target grid: `1.351s/step`, `0.705s` backward,
  `0.041s` target/loss, `1.0MiB` resident target grid.
- Selected STAR RGB feature diagnostic: `2.491s/step`, `1.184s` backward.
- Gaussian/token recon-only cached conditioning: `3.460s/step`,
  `1.963s` backward.
- Gaussian/token prediction-side V-JEPA loss: `38.621s/step`,
  `36.762s` backward.

## Interpretation

Target-grid is now the fastest STAR V-JEPA target diagnostic and removes the
cached-chunks memory cliff. It is not a dense render-grid-loss replacement yet:
the objective changes from dense `[T,F,H,W]` MSE to MSE on the V-JEPA token
grid after downsampling rendered features. That may be the right semantic
objective, but it needs a longer media/quality gate before promotion.

`cached_chunks` remains the exact dense render-grid-loss reference for short
runs. The target-cache budget says dense float32 adapted targets are already
`2GiB` at 64f/512px/F32, `4GiB` at 128f/512px/F32 or 64f/512px/F64, and `8GiB`
at 64f/1024px/F32.

## Next Gates

1. Run a longer target-grid media/quality overfit on the same test video to see
   whether the coarse target-grid loss produces useful visual behavior.
2. Prototype a native VJP or loss handoff if we need dense render-grid MSE
   without resident multi-GiB adapted targets.
3. Keep the selected `star-feature-512-fast` route labeled as RGB-target
   `FeatureToColor` training, not precomputed V-JEPA training.
4. Do not scale this to the 300-set until the target-grid quality gate and the
   Gaussian/token 512px NaN guardrails are clear.

## Validation

- `py_compile` on the STAR target trainer passed before the benchmark.
- `tests/test_star_uvt_feature_target_adapter.py` passed with 4 tests.
- The target-grid 5-step benchmark passed and regenerated the comparison,
  bridge-audit, and target-cache-budget reports.
