# STAR UVT Batched Trainer Status

## Original Goal

The broad request was to stop guessing about STAR UVT speed and rebuild the
working plan from evidence:

- rerun and audit STAR UVT / dynamic gsplat shader benchmarks across frame
  counts and resolutions
- break down forward and backward time, especially whether backward was
  renderer time or feed-forward/colorizer/loss time
- recover the fast sparse/bounding-box feature-splatting tricks and port the
  useful parts into STAR UVT
- choose the best UVT STAR shader path for a fast single-video overfit
- identify real training bottlenecks, then scale the chosen path toward the
  prepared 300-video set
- keep WorldFoam/Gate4 as a side investigation without letting it fight for
  GPU/trainer time

The immediate closeout goal for this chunk was narrower: repeat the core plan
docs, record any missing details, execute the next benchmark gate, and record
progress in markdown as each step landed.

## Current State

The selected 64f/512px/8192-tube target-grid/frozen-probe STAR UVT path is now:

```text
feature_target.image_vjp_mode = analytic_sparse_grid_forward_batched
feature_uvt.render_mode = feature_direct_gradcache_reduce_vec4
target materialization = target_grid
source checkpoint = 1300-step feature/probe checkpoint
```

What changed:

- sparse feature forward still renders only the `65,536` target-support pixels
  instead of the dense 512px feature image
- target-grid feature loss, frozen hidden64 RGB-probe loss, and sparse-grid VJP
  are now batched across all 32 frame chunks in one MPS path
- the batched path is integrated into `src/train/train_star_uvt_feature_overfit.py`
  as the opt-in mode `analytic_sparse_grid_forward_batched`
- the checked config is
  `src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume5_from1300_sparseforward_batchedvjp.jsonc`

Key benchmark results:

```text
isolated target/probe loss+VJP: 38.028ms -> 4.759ms (7.990x)
loss error: 7.45e-09
max feature grad error: 6.55e-11

5-step harness no-first step: 173.139ms
5-step harness no-first render: 71.297ms
5-step harness no-first batched loss+VJP: 7.264ms
5-step harness no-first backward: 67.395ms

first-class trainer single run:
  pass true, zero overflow
  loss 0.886537 -> 0.885009
  feature loss 0.632124 -> 0.631692
  frozen RGB-probe PSNR 21.965 -> 21.984
  no-first step 226.595ms
  no-first backward 92.052ms

first-class trainer repeat-3:
  pass true for all rows, zero overflow
  no-first step mean/min/max/stdev 179.304/159.658/215.613/31.480ms
  no-first backward mean/min/max/stdev 71.999/60.800/90.170/15.878ms
  no-first render mean/min/max/stdev 71.115/67.791/77.420/5.463ms
```

This supersedes the older sparse-forward repeat-3 comparison surface:

```text
old analytic_sparse_grid_forward no-first step: 504.9/411.0/626.4/110.3ms
old analytic_sparse_grid_forward no-first backward: 142.2/114.7/174.4/30.1ms
```

## Recorded Artifacts

Canonical docs updated:

- `README.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `PROJECT_INDEX.md`
- `BASELINES.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`

Benchmark/code artifacts:

- `research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py`
- `research_experiments/star_uvt_feature_tubes/sparse_forward_batched_step_benchmark.py`
- `research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py`
- `outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_target_vjp_profile.md`
- `outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batched_step_benchmark.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_64f512_from1300_5step.json`
- `outputs/benchmarks/2026-05-19_star_uvt_sparse_forward_batchedvjp_512_repeat3_timing.md`

## What Is Left

1. Run a longer single-video overfit with the batched trainer mode and media so
   speed does not mask a quality regression.
2. Promote the fast script route to call the batched target-grid/frozen-probe
   config when the goal is UVT STAR feature-tube overfit speed.
3. Recheck dynamic gsplat / Gaussian baselines at matched frame count,
   resolution, and splat count using the same benchmark table shape.
4. Scale from the single-video checkpoint gate to the prepared 300-video set
   only after the longer overfit remains stable.
5. If another shader pass is justified, beat the batched repeat distribution;
   dense VJP packing and dense-forward variants are already negative, and
   `feature_direct_fixedbin` is still only a direct-atomic alias until a real
   fixedbin/tile-slot kernel exists.
6. Keep WorldFoam/Gate4 separate: continue note-writing and shader probes
   without tying it to the main GPU/trainer lane.
