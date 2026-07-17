# STAR UVT doc sync and remaining plan

Date: 2026-05-19 17:59

## Original goal

Repeat and harden the STAR UVT fast feature-shader plan docs, fill any missing
implementation details, then execute the plan gate by gate with benchmarks and
progress logs recorded in markdown.

The concrete work plan from the handoff was:

1. Re-run the STAR UVT / dynamic GSplat shader benchmark matrix at matched
   frame counts, splat/tube counts, and resolutions, recording forward/backward
   timing.
2. Decide which shaders are real keeper paths versus diagnostics.
3. Build a fast single-video overfit route for the selected STAR UVT feature
   path and the dynamic-GSplat path.
4. Break down the true training bottlenecks.
5. Scale only the keeper path to the prepared 300-video dataset.
6. Carry the feature-splatting lessons forward into UVT feature world tubes.
7. Keep WorldFoam shader investigation separate so it does not fight for GPU.

## Current state

The current selected STAR UVT cached-V-JEPA speed diagnostic is
`star-feature-512-fast`: `feature_target.image_vjp_mode =
analytic_sparse_grid_forward_batched`, `feature_direct_gradcache_reduce_vec4`,
64 frames, 512px, 8192 tubes, F32, resumed from the 1300-step checkpoint.

The lr005 sparse-forward batched-VJP 100-step media gate passes and preserves
the older dense objective movement while cutting mean step/backward/render to
`399.9/176.9/125.2ms` and last-20 step to `262.9ms`. The effective-lr001 rerun
also passes and preserves the dense lr001 endpoint at feature loss `0.630549`
and probe PSNR `22.034`, while cutting mean step/backward/render to
`372.3/158.9/119.9ms`. It is not a quality promotion over lr005 because lr005
still ends with better feature and weighted loss, and the lr001 late timing
window is noisy.

The important diagnostic shift is that speed is no longer the main excuse for
the target-grid/frozen-probe path. Sparse-grid VJP removed the dense image-VJP
packing, sparse-forward removed dense feature-image rendering for the target
grid support, and batched target/probe VJP removed most per-chunk Python/VJP
overhead. The remaining blocker is visual/objective quality: the probe media is
still blurry, and the STAR feature route still has not closed the same-grid
feature-to-RGB oracle or the RGB STAR source-view quality bracket.

Docs now updated or cross-linked:

- `README.md`
- `PROJECT_INDEX.md`
- `TODO/README.md`
- `EXPERIMENTS.md`
- `BASELINES.md`
- `research_experiments/star_uvt_feature_tubes/README.md`
- `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
- `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py`
- `agent_notes/key_learnings.md`

Primary fresh evidence:

- `agent_notes/loose_notes/2026-05-19_17-53-03_star_uvt_lr001_sparse_batched_gate.md`
- `outputs/benchmarks/2026-05-19_star_uvt_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.md`
- `outputs/benchmarks/2026-05-19_star_uvt_feature_targetgrid_sparseforward_batchedvjp_lr001_64f512_from1300_100step_media.json`
- `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`

## What is left

1. Validate the doc sync and benchmark invariants after this note.
2. Stop spending time on tile-overflow or checkpoint-plumbing loops for this
   lane unless a fresh benchmark contradicts the current read.
3. Choose the next quality gate: checkpoint selection, a recovery schedule keyed
   to measured transients, or a stronger objective bridge toward the frozen
   feature-to-RGB oracle.
4. Only pursue native GPU target/probe VJP or real scalar fixedbin/tile-slot
   feature-gradient work if it beats the current sparse-forward batched-VJP
   speed surface.
5. After a quality gate produces non-blurry media, run the same selected route
   through the fast overfit script and then scale to the prepared 300-video set.
