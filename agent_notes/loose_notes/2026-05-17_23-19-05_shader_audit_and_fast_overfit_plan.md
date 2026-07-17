# Shader Audit And Fast Overfit Plan

Date: 2026-05-17
Workspace: `/Users/nicholasbardy/git/gsplats_browser/dynaworld`

## User Goal

Audit current dynamic gsplat and STAR UVT shader/training paths, benchmark
forward/backward across frame counts and resolutions, pick usable fast overfit
scripts, identify real bottlenecks, plan the 300-set scale-up, and keep separate
feature-tube and WorldFoam investigations from fighting for the GPU.

## Direct Answers

- The 300-clip multires precomputed V-JEPA path exists for the Gaussian trainer,
  not STAR UVT:
  `src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc`.
- That config uses cached V-JEPA conditioning and disables differentiable
  prediction-side V-JEPA feature loss. It is RGB Gaussian splatting, not F32
  feature splatting.
- The 300-record V-JEPA feature cache is complete: `300/300` cache files in
  `data/feature_cache/single_video_pretrain_300_youtube_vjepa2_1_vitb_256crop_64f_512center_nativefps`.
- The Gaussian trainer still does not use `torch.utils.data.DataLoader`; it uses
  a bounded `ThreadPoolExecutor` sequence prefetch when
  `data.train_manifest_prefetch > 0`. The 300-clip multires config sets
  `train_manifest_prefetch=2`.
- A 12-step 300-record scale probe confirmed the path is cache-hot and prefetch
  is active. Warm timing at the 256px stage is about `2.06s/step`, with
  backward still the largest bucket.
- STAR UVT uses the direct-atomic speed valve too: current practical route is
  `sample_emission_mode=direct_atomic`, `reduction_mode=index_add`.
- STAR UVT is overfitting in the intended source-view sense. The 512px
  first-class run reached PSNR `29.138` and SSIM mean/min `0.8606 / 0.7794` on
  the high-motion 64-frame clip.

## New First-Class 512px STAR UVT Result

Config:

```text
src/train_configs/star_uvt_highmotion_hlaZbH_64f_512_directatomic_multires256c200_50fine.jsonc
```

Run:

```text
W&B: https://wandb.ai/nbardy/dynaworld/runs/4r2x8s3c
result JSON: star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.json
contact sheet: star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.png
side-by-side MP4: star_uvt/results/may17_firstclass_highmotion_hlaZbH_start2_512_64f_32768t_cap256_directatomic_multires256c200_50fine.mp4
```

Metrics:

```text
coarse: 256px, 200 steps, 119.14s, final loss 0.0010415
fine: 512px, 50 steps
total UVT wall: 232.73s
final loss/MSE: 0.0012195
L1: 0.01777
PSNR: 29.138
SSIM mean/min/max: 0.8606 / 0.7794 / 0.9538
render benchmark median: 393.20ms
tile_load_final_proxy: 1484.35
```

Interpretation:

```text
This first-class route reproduces the older harness 512px multires quality
almost exactly. It is the 512px STAR UVT row to use for high-motion source-view
overfit. The wall clock is slower than the older harness note, so do not claim a
new speed best; claim first-class reproducibility, W&B/media, and quality.
```

## Shader Timing Snapshot

Artifacts live under:

```text
outputs/benchmarks/2026-05-17_shader_audit/
```

Dynamic gsplat projected forward+backward, synthetic projected splats:

| case | fastest row | median/mean signal |
| --- | --- | --- |
| 256px, 8192 splats, B=16 | v5 | sparse 48.8ms, medium 83.4ms |
| 512px, 8192 splats, B=16 | v5 | sparse 56.4ms, medium 111.2ms |
| 256px, 8192 splats, B=64 | split | v5 sparse 178.4ms, v6 medium 314.0ms |
| 512px, 8192 splats, B=64 | split | v5 sparse 312.9ms, v6 medium 525.6ms |
| 512px, 32768 splats, B=16 | split | v5 sparse 181.1ms, v6 medium 363.3ms |

Takeaway:

```text
There is no single gsplat shader winner. v5 is stronger for small/sparse rows
and current low-overhead training; v6 wins some wider/medium-footprint rows. The
current 300-clip config uses v6_refined RGB, which is defensible for larger
learned support, but the fast overfit script should keep profiling enabled and
not assume raster is the only bottleneck.
```

STAR UVT train-step timing, high-motion clip:

| case | mode | median total | median forward | median backward |
| --- | --- | ---: | ---: | ---: |
| 256px, 16f, 32768 tubes | direct_atomic/index_add | 178.8ms | 21.5ms | 150.8ms |
| 256px, 64f, 32768 tubes | direct_atomic/index_add | 386.5ms | 38.0ms | 324.1ms |
| 512px, 16f, 32768 tubes | direct_atomic/index_add | 267.8ms | 25.3ms | 232.4ms |
| 512px, 64f, 32768 tubes | direct_atomic/index_add | 1288.7ms | 139.9ms | 1096.2ms |
| 256px, 64f, 8192 tubes | direct_atomic/index_add | 172.0ms | 9.9ms | 154.1ms |
| 256px, 64f, 8192 tubes | tile_pair_suffix/keyseg | 232.3ms | 7.7ms | 218.7ms |

Takeaway:

```text
Direct atomic is the practical STAR UVT overfit branch. At 512px/64f/32768
tubes, backward dominates. Compact deterministic can look close on tiny early
steps at 8192 tubes, but prior 50-step evidence says it still blows up when
tile load grows, so it remains a blocker lane rather than the training default.
```

## Gaussian Trainer Bottlenecks

Two short probes were run through `train_single_video_pretrain_300_64f.sh`.

RGB Gaussian, 512px, 512 splats/frame, 300 records, cached V-JEPA conditioning:

```text
Timing step 5:
step_total=5.3925s
sample_clip=1.3013s
forward_decode=0.7583s
render/rasterize=0.1973s
loss/reconstruction=0.3399s
backward=2.7200s
optimizer_step=0.0047s
```

RGB Gaussian, multires overfit first record, 8192 splats/frame, current 256px
stage:

```text
Timing step 5:
step_total=3.9265s
sample_clip=0.7070s
forward_decode=0.6697s
render/rasterize=0.1731s
loss/reconstruction=0.0665s
backward=2.2782s
optimizer_step=0.0078s
```

Takeaway:

```text
The 300-set is cache-hot, so the old V-JEPA extraction issue is gone. The next
Gaussian bottlenecks are model/backward and synchronous clip loading, not only
raster. Prefetch should help sample_clip on the full 300-record run, but a
full DataLoader rewrite is not proven necessary yet.
```

300-record multires scale probe, 8192 splats/frame, 256px warm stage,
`train_manifest_prefetch=2`, cached V-JEPA conditioning:

```text
command log:
outputs/run_logs/scale300_prefetch2_8192_multires_profile12_20260517_232500.log

train records: 300
feature cache: hit per sampled record
prefetch: enabled, depth=2
render schedule: 256px until step 2400, then 512px

Timing step 5:
step_total=2.0918s
sample_clip=0.1717s
forward_decode=0.4432s
render/rasterize=0.1290s
loss/reconstruction=0.0667s
backward=1.2571s
optimizer_step=0.0020s

Timing step 10:
step_total=2.0561s
sample_clip=0.0388s
forward_decode=0.4392s
render/rasterize=0.1344s
loss/reconstruction=0.0727s
backward=1.3437s
optimizer_step=0.0019s
```

Interpretation:

```text
The full 300-record route is ready for a monitored scale run. Data loading is
not currently the first bottleneck once the prefetch queue is warm. At 256px,
backward is roughly 65% of warm step time, forward decode is roughly 21%, and
raster is roughly 7%. The main unknown remains the scheduled 512px promotion at
step 2400, because the earlier 300-record Gaussian run had a post-promotion NaN
failure.
```

## Scripts Added

Shader audit runner:

```text
src/train_scripts/benchmark_star_uvt_gsplat_shaders.sh
```

Selected fast overfit runner:

```text
src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh
```

Useful commands:

```bash
./src/train_scripts/benchmark_star_uvt_gsplat_shaders.sh
./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-512
./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh gsplat-smoke
./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh gsplat-overfit
```

## Validation

Completed after the benchmark/config/note updates:

```text
rtk git diff --check
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_star_uvt_video_overfit.py \
  research_experiments/star_uvt_feature_tubes/dense_feature_tube_prototype.py
rtk ./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh help
rtk env RUN_GSPLAT=0 RUN_UVT=0 OUT_DIR=/tmp/dynaworld_shader_audit_noop \
  ./src/train_scripts/benchmark_star_uvt_gsplat_shaders.sh
rtk env PYTHONPATH=src/train .venv/bin/python - <<'PY'
from config_utils import load_config_file
from pathlib import Path
for path in [
    'src/train_configs/star_uvt_highmotion_hlaZbH_64f_512_directatomic_multires256c200_50fine.jsonc',
    'src/train_configs/local_mac_single_video_pretrain_300_youtube_64f_512render_static_dynamic_register_recon_only_gpt256_8192splats_300clips_3kstep_multires_256to512.jsonc',
]:
    cfg = load_config_file(Path(path))
    print(path, cfg.get('arch'), cfg.get('train', {}).get('steps'))
PY
```

All completed successfully.

## Subagent Results

Feature tubes:

```text
agent_notes/loose_notes/2026-05-17_23-12-03_star_uvt_feature_tubes_investigation.md
research_experiments/star_uvt_feature_tubes/
```

Conclusion: current STAR UVT is RGB-only. F32 feature tubes need a feature
renderer fork with `[N,F]` features, feature image output, feature gradients,
and external image-space colorization first. The isolated dense CPU prototype
proved gradients reach tube features and centers, but this is not yet a GPU
trainer.

WorldFoam:

```text
agent_notes/loose_notes/2026-05-17_23-08-34_world_foam_side_investigation_status.md
```

Conclusion: current WorldFoam gate is still narrow fixed-geometry RGB replay.
The next useful shader is a Metal CSR compositor/VJP that consumes the Gate 4
affine moving-ray slab tape directly; do not keep mutating the promoted
framegroup16 op for the same bottleneck.

## Next Plan

1. Use STAR UVT direct_atomic/index_add and the first-class 512px multires
   config as the current high-motion source-view overfit lane.
2. Use the dynamic gsplat multires overfit config as the Gaussian comparison,
   but keep `profile_timing=true` and inspect `backward` plus `sample_clip`;
   do not assume the rasterizer is the limiting term.
3. Scale to the 300-set with the cache-hot multires config and
   `train_manifest_prefetch=2`; monitor the step-2400 512px switch because the
   prior Gaussian 300-run NaN'd after promotion. Use the 12-step profile above
   as the local throughput baseline before starting a long run.
4. For STAR UVT speed, target high-res backward and tile-load growth. The
   forward pass is not the dominant 512px/64f term.
5. For deterministic STAR UVT, fix load-growth/backward in the
   `tile_pair_suffix` / keyed segmented path before trying larger quality runs.
6. For feature UVT, keep the dense prototype isolated until a minimal feature
   Metal direct-atomic fork exists.
