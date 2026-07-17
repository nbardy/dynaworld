# STAR UVT Fast-Shader Goal Audit

Date: 2026-05-20 01:23 +07

## Objective Under Audit

Active goal:

> Repeat and harden the STAR UVT fast feature-shader plan docs, fill any
> missing implementation details, then execute the plan gate by gate with
> benchmarks and progress logs recorded in markdown.

The broader handoff plan also included matched dynamic-gsplat timing,
single-video overfit route selection, bottleneck diagnosis, scale-up to the
prepared 300-video set, feature-splatting carry-forward, and a separate
WorldFoam lane.

## Requirement Audit

| Requirement | Evidence | Status |
| --- | --- | --- |
| Harden STAR UVT fast-shader plan docs | `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`, `research_experiments/star_uvt_feature_tubes/README.md`, `EXPERIMENTS.md`, `PROJECT_INDEX.md`, `README.md`, `TODO/README.md`, `BASELINES.md` all now route to the selected helper, native target-area baselines, compact visual route, and rejected diagnostics. | Complete for the STAR UVT shader diagnostic phase |
| Fill missing implementation details | Native target-area modes, colorizer-gradient modes, Torch reducer prototype, and SIMD-reduced colorizer modes are documented in reports and source. The latest report records mode bits, Python mode names, trainer mode names, build command, parity gates, timing, and rejection reason. | Complete for the tested gates |
| Execute gates with benchmarks | Reports exist for selected-shader scale, cached V-JEPA bridge, sparse-pixel/grid/batched VJP, compact visual route, full-cell native target-area, hidden-size/split/W^T diagnostics, colorizer-gradient ABI, atomic split, Torch reducer prototype, and SIMD-reduced colorizer. The latest SIMD-reduce report records direct-kernel and trainer evidence. | Complete for the shader plan gates run so far |
| Record progress logs in markdown | Each major gate has a loose note under `agent_notes/loose_notes/`, including `2026-05-20_01-18-22_star_uvt_colorizer_simdreduce_gate.md` and the follow-up `2026-05-20_01-48-12_star_uvt_rgb_grid_lowfreq_bridge_gate.md`. Benchmark reports live under `outputs/benchmarks/`. | Complete for the shader plan gates run so far |
| Keep dense key learnings compressed | `agent_notes/key_learnings.md` remains under the 200-line cap and includes the compact target-area/native colorizer lesson on line 199. | Complete |
| Identify current practical route | Docs consistently select `star-feature-512-visual` / compact autograd as the practical visual route, with full-cell8 native vec4 W^T as exact full-support baseline. SIMD-reduce is recorded as a direct-kernel win but trainer reject. | Complete |
| Validate docs/code after edits | Current validation passed: `bash -n` for the helper script, `py_compile` for touched STAR UVT trainer/benchmark/wrapper files, `wc -l` for key learnings, root `git diff --check`, third-party `git diff --check`, and trailing-whitespace scan on updated markdown files. | Complete |
| Scale to prepared 300-video set | Existing docs describe this as a next phase after non-blurry single-video quality. The selected visual-quality gate now explicitly fails scale-up: `outputs/benchmarks/2026-05-20_star_uvt_selected_visual_quality_gate.md` records dense full RGB `6.023` PSNR, sparse/streaked or blurry media, and RGB STAR bracket `12.444` PSNR. No new 300-video STAR UVT scale run was launched. | Not complete; blocked by visual quality |
| Matched dynamic-gsplat rerun at same config | Smoke-level rerun now exists: `outputs/benchmarks/2026-05-20_dynamic_gsplat_512_matched_probe.md` records fixed `64f/512px/8192` active-Gaussian step-5 timing at `8.019s` total / `5.638s` backward. This proves a fresh local comparator, but not a final dynamic-gsplat quality/ranking baseline. | Smoke complete; full baseline still open |
| Feature-splatting carry-forward / WorldFoam side lanes | Notes exist and lanes are kept separate, but this STAR UVT shader closeout did not execute new work in those lanes. | Not complete |

## Decision

The STAR UVT fast feature-shader diagnostic phase is complete enough to stop
micro-optimizing the colorizer/native target-area route:

- `star-feature-512-visual` remains the practical single-video visual helper.
- Full-cell8 native vec4 W^T remains the exact full-support native baseline.
- Naive native colorizer gradients are correct but too slow.
- Torch sidecar colorizer reduction is correct but loses to sparse-pixel.
- Same-pass SIMD-reduced colorizer fixes the direct native atomic envelope
  (`297.2ms` native compact total versus `312.1ms` sparse-pixel baseline in the
  matched direct gate), but the trainer still rejects it (`2908.9ms` mean step,
  `604.0ms` sparse visual backward, same feature/probe regression).
- The first trainable low-frequency RGB-grid bridge is also rejected: it is fast
  (`353.1ms` mean step, `289.9ms` no-first) and improves the grid metric
  (`22.028 -> 22.248` PSNR), but feature loss worsens and dense RGB falls to
  `5.657` PSNR with sparse/streaked media.

Do not mark the broader active goal complete yet if the intended success
includes the original scale-up and matched dynamic-gsplat work. The next useful
phase should be explicitly scoped as visual-quality/scale execution, not more
colorizer-atomic shader work.

## Next Work

1. Change the visual objective/support/model bridge and rerun the selected
   visual-quality gate until dense media is no longer sparse/streaked or blurry.
2. If a stronger dynamic-gsplat claim is needed, run repeat timing rows and
   media/W&B for the fixed-512 or guarded multires config; the smoke already
   shows the current dynamic-gsplat path is not the fast local route.
3. If single-video quality clears the visual gate, launch the prepared
   300-video scale lane with W&B, checkpoints, media, and a new `BASELINES.md`
   row.
4. Keep feature-splatting and WorldFoam as separate documented lanes so they do
   not consume the main STAR UVT benchmark GPU window.
