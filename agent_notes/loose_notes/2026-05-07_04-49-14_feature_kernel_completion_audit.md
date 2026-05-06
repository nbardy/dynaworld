# Feature Kernel Completion Audit

## Objective Restatement

The active thread objective is to iterate on copied Metal feature-splatting
kernels, benchmark them on larger batches and feature dimensions, drive raster
speed down without freezing the machine, and leave durable shared notes about
Metal shader performance, bottlenecks, safety, and next steps.

This audit does not mark the objective complete. It records what is currently
proved by files and commands, and what still needs work before a fork should be
promoted.

## Prompt-To-Artifact Checklist

| Requirement | Evidence | Status |
| --- | --- | --- |
| Preserve the stable working kernel | `git -C third_party/fast-mac-gsplat diff --stat -- variants/v6_refined_features` and `diff --name-only` both returned empty output. | Covered |
| Fork new kernel versions instead of mutating baseline | New untracked variant dirs exist for `v6_refined_features_f32_reduce`, `f32_accum`, `f32_gradcache`, `f32_fixedbin`, `f32_block4`, `f64_accum64`, `f32_stage`, and `v6_feature_lookup_experiment`; no diff under stable `v6_refined_features`. | Covered |
| Keep experimental forks opt-in | `src/train/renderers/fast_mac.py` dispatches by explicit `render.fast_mac.feature_variant`; checked-in configs currently reference only `v5_features` or stable `v6_refined_features`. | Covered |
| Benchmark bigger batches/features | `research_notes/fast_mac_feature_metal_performance.md` cites local artifacts for `B=16/B32`, `F=32/F64`, `128/256/512px`, plus trainer fixed-render paths. A cited-artifact existence check over that note and the loose note produced no `MISSING` lines. | Covered as local timing evidence |
| Drive down speed | Measured wins exist: `f32_reduce` cuts stable `512px/B16/G8192/F32` raster total from `1001.9ms` to `449.2ms`; `f32_accum` reaches `388.1ms` in one corrected-cap row; `f32_fixedbin` reaches `501.8ms` in the later target row versus `855.4/706.5/716.4ms` same-window alternatives. | Partially covered; shape-dependent |
| Trainer-path timing isolation | `src/benchmarks/trainer_phase_benchmark.py` has `--seed`, `--fixed-render-graph`, and frozen-color fixed-render mode; artifacts under `benchmark_outputs/trainer_phase/` compare stable and forks. | Covered |
| Trainer-path correctness before long runs | `src/benchmarks/fixed_render_variant_parity.py` exists and artifacts show 256px train/heldout exact forward parity plus 128px train/heldout gradient parity for `f32_reduce`, `f32_accum`, `f32_gradcache`, and `f32_fixedbin`. | Covered as pre-flight parity |
| Write shared Metal shader performance docs | `research_notes/fast_mac_feature_metal_performance.md` covers artifact map, timing tables, Metal performance model, bottlenecks, safe benchmark contract, profiling workflow, and next forks. | Covered |
| Write raw agent notes | `agent_notes/loose_notes/2026-05-07_02-15-01_feature_kernel_fork_iteration.md` records forks, commands, validation, benchmarks, interpretation, and safety notes. | Covered |
| Update key learnings with surprising results | `agent_notes/key_learnings.md` remains under 200 lines and now records reduction, cap/fallback, staging, gradcache, fixedbin, and trainer fixed-render lessons. | Covered |
| Be careful not to freeze the machine | Benchmark notes constrain safe shapes, forbid 4K/64K and overflow-stress sweeps, use sequential runner and timeouts, and record the largest shapes tried locally. | Covered for current work |

## Current Verified Boundaries

- Stable baseline untouched: no diff under `variants/v6_refined_features`.
- Checked-in configs are not silently switched to experimental forks.
- Experimental trainer dispatch is opt-in by string, not default behavior.
- All benchmark and parity artifacts cited by the shared notes currently exist
  on disk.
- Root and submodule `diff --check` passed after the docs update.
- `py_compile` passed for the touched Python dispatch and benchmark files.

## Bugs Or Risks Introduced

- `f32_fixedbin` is no-overflow only. Its Python wrapper raises when a tile
  exceeds `max_fast_pairs`; that is intentional, but it means it is not a drop-in
  replacement for overflow-capable forks.
- The fixed-bin path trades a sync/allocation for a larger fixed ID buffer
  (`~128 MiB` at `512px/B16/tile16/cap2048`). That can hurt memory pressure even
  when timing wins one target row.
- Same-session timing remains noisy. The strongest fork changes between rows:
  `f32_reduce` is simplest/proven, `f32_accum` wins some corrected-cap rows,
  `f32_gradcache` wins one fixed-render window, and `f32_fixedbin` wins the
  later target synthetic row. None should be default yet.
- The parity checks prove fixed render/loss equivalence, not optimizer-time
  heldout quality.

## Missing Before Promotion

1. Run a real W&B train/heldout-quality parity job before promoting any fork.
2. Capture hardware counters with Xcode/Metal for the likely finalist shapes:
   `512px/B16/F32`, `256px/B32/F32`, and a bounded `F64` row.
3. Add an overflow-aware fixedbin sibling only if profiles show fixedbin wins
   enough to justify supporting overflow instead of fail-fast.
4. Test trainer microbatch/framewise backward as the non-kernel lever for dense
   `[B,H,W,F]` memory pressure.
5. Commit the submodule fork dirs first, then the dynaworld root pointer/docs,
   if this work is ready to publish.

## Audit Conclusion

The current state is a strong local benchmark and documentation checkpoint, but
the larger goal is not complete. The next decision should be either a scoped
commit of this checkpoint or one real train/heldout-quality run using the best
opt-in candidate.
