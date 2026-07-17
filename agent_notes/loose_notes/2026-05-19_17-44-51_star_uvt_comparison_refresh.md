# STAR UVT Comparison Refresh

Date: 2026-05-19 17:44 Asia/Ho_Chi_Minh

## Original goal

Repeat and harden the STAR UVT fast feature-shader plan docs, fill any missing
implementation details, then execute the plan gate by gate with benchmarks and
progress logs recorded in markdown.

Expanded working goal from the thread:

1. Re-run and audit the available Gaussian/dynamic/STAR UVT shader paths at
   matched frame counts, splat/tube counts, and resolutions, with forward and
   backward timing.
2. Decide what renderer/shader path is good enough to use versus what still
   needs fixing.
3. Build a fast single-video overfit path for the selected STAR UVT feature
   shader and the dynamic Gaussian baseline.
4. Identify the real training bottlenecks instead of assuming the rasterizer is
   always the problem.
5. Scale the selected route toward the prepared 300-video dataset after the
   single-video gate is coherent.
6. Keep the feature-splatting and WorldFoam side investigations documented
   without letting them fight the main GPU path.

## Current state

The selected STAR UVT helper route is now:

```bash
PYTHONPATH=src/train ./src/train_scripts/train_fast_overfit_star_uvt_and_dynamic_gsplat.sh star-feature-512-fast
```

That helper now uses the cached V-JEPA target-grid/frozen-probe sparse-forward
batched-VJP route:

```text
src/train_configs/star_uvt_feature_testvideo_64f_512_vjepa_target_gradcache_reduce_vec4_chunk2_8192t_no_prenorm_targetgrid_rgbprobe40_feature1_lr005_resume100_from1300_sparseforward_batchedvjp_checkpoint_media.jsonc
```

The old RGB-target speed diagnostic is preserved as `star-feature-512-rgbfast`.

## Evidence refreshed

The comparison generator now includes the sparse-forward batched-VJP 100-step
helper/media row in:

```text
outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md
outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.json
```

Matched 64f/512px/8192 row summary:

- old dense target-grid/frozen-probe 1300->1400: `1.690s/step`,
  `0.910s` backward, `0.617s` render
- sparse-forward batched-VJP helper/media 1300->1400: `0.400s/step`,
  `0.177s` backward, `0.125s` render
- sparse-forward batched-VJP last-20: `0.263s/step`, `0.109s` backward,
  `0.094s` render
- objective movement is preserved: feature loss `0.632124 -> 0.627122`,
  frozen-probe PSNR `21.965 -> 21.979`, zero overflow
- same-grid visual oracle still wins: target-grid feature-to-RGB probe reaches
  `23.401` grid PSNR, while the STAR probe media is still blurry

Matched context:

- selected RGB feature diagnostic: `2.491s/step`, `1.184s` backward
- Gaussian/token recon-only cached conditioning: `3.460s/step`, `1.963s`
  backward
- Gaussian/token prediction-side V-JEPA loss: `38.621s/step`, `36.762s`
  backward

## Docs updated

Updated the canonical summaries so future agents do not have to infer the
current speed route from loose notes:

- `README.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `PROJECT_INDEX.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `agent_notes/key_learnings.md`

## What is left

The speed path is now good enough for the single-video cached V-JEPA target
diagnostic. The remaining blocker is quality, not basic target-route speed:

- close the gap to the same-grid `23.401` PSNR oracle with checkpoint selection,
  better objective balance, or a measured-recovery schedule
- run the selected helper at the next dataset scale only after the visual
  diagnostic stops producing blurry media
- only pursue native GPU target/probe VJP or real fixedbin if it beats the
  sparse-forward batched-VJP speed surface
- keep Gaussian 300-set comparisons separate: recon-only cached conditioning is
  the useful timing reference; prediction-side frozen V-JEPA loss is the known
  backward-dominated negative control
