# STAR UVT Batched Target-Grid VJP Gate

## Goal

Continue the STAR UVT sparse-forward speed plan after the repeat-3 512px gate.
The specific question was whether the remaining target-grid/frozen-probe loss
and sparse VJP work is just per-chunk Torch launch overhead before spending on
a lower-level native target-grid/probe loss+VJP shader.

## Artifacts

```text
research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py
research_experiments/star_uvt_feature_tubes/sparse_forward_batched_step_benchmark.py
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_target_vjp_profile.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_target_vjp_profile.json
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_step_benchmark.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_step_benchmark.json
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforward_batchedvjp.jsonc
outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_5step.json
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batchedvjp_512_repeat3_timing.md
outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batchedvjp_512_repeat3_timing.json
```

## Commands

```bash
PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py \
  --warmup 1 --repeat 3 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_target_vjp_profile

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_forward_batched_step_benchmark.py \
  --steps 5 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_step_benchmark

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python src/train/train_star_uvt_feature_overfit.py \
  src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforward_batchedvjp.jsonc

PYTHONPATH=src/train:third_party/fast-mac-gsplat/variants/star_uvt_v0 \
  .venv/bin/python research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py \
  --base-config src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforward_batchedvjp.jsonc \
  --repeat 3 --timeout-sec 300 \
  --out-base outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batchedvjp_512_repeat3_timing
```

## Result

The isolated batched target/probe VJP profile is positive after warmup:

```text
per-chunk total_loss_vjp_ms: 38.028
batched total_loss_vjp_ms: 4.759
speedup: 7.990x
feature_target_ms: 11.276 -> 2.680
rgb_probe_loss_ms: 17.301 -> 1.152
image_vjp_ms: 9.451 -> 0.928
loss error: 7.45e-09
max feature grad error: 6.55e-11
pixel mismatches: 0
tile overflow: 0
```

The full 5-step optimizer harness is also positive:

```text
pass: true
loss: 0.886537 -> 0.885759
feature loss: 0.632124 -> 0.631763
rgb-probe loss: 0.006360 -> 0.006350
no_first_step_ms: 173.139
no_first_render_forward_ms: 71.297
no_first_batched_loss_vjp_wall_ms: 7.264
no_first_backward_ms: 67.395
no_first_renderer_backward_ms: 41.621
no_first_param_backward_ms: 25.773
zero overflow, max tile 63
```

The first-class trainer mode is also positive:

```text
feature_target.image_vjp_mode: analytic_sparse_grid_forward_batched
pass: true
loss: 0.886537 -> 0.885009
feature loss: 0.632124 -> 0.631692
rgb-probe PSNR: 21.965 -> 21.984
single trainer no_first_step_ms: 226.595
single trainer no_first_backward_ms: 92.052
repeat-3 no_first_step mean/min/max/stdev: 179.304/159.658/215.613/31.480
repeat-3 no_first_backward mean/min/max/stdev: 71.999/60.800/90.170/15.878
repeat-3 no_first_render mean/min/max/stdev: 71.115/67.791/77.420/5.463
zero overflow, max tile 63, p95 tile 42
```

## Interpretation

This is not a native shader yet, but it is now a first-class trainer mode. The
preflight harness proved that batching the tiny target-grid/probe work across
all 32 chunks is a real lever, and the trainer repeat confirms the speedup
survives first-class optimizer-state resume. The old repeat-3 sparse-forward
trainer comparison surface was:

```text
no_first_step mean/min/max/stdev: 504.9/411.0/626.4/110.3ms
no_first_backward mean/min/max/stdev: 142.2/114.7/174.4/30.1ms
```

The batched trainer repeat is below that distribution with the same pass
criteria: same loss/probe movement, zero overflow, and positive model
gradients. It should be treated as the selected 512px target-grid/frozen-probe
speed surface for the next overfit pass.

Native GPU target-grid/probe loss+VJP remains justified only if it can beat the
batched trainer repeat distribution, or if longer overfit shows that the
trainer path is quality-limited rather than simply faster.
