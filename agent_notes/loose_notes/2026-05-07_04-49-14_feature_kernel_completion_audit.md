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
| Trainer-path timing isolation | `src/benchmarks/trainer_phase_benchmark.py` has `--seed`, `--fixed-render-graph`, frozen-color fixed-render mode, and opt-in `--memory-sample-interval-ms`; the cited trainer-phase JSON artifacts compare stable and forks. | Covered |
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

## Post-Audit Update

After this audit, the checkpoint was committed and the lookup prototype received
two additional gates:

- `v6_feature_lookup_experiment/tests/feature_lookup_parity_check.py` now checks
  compact-basis direct-vs-lookup parity for features, alpha, loss, and gradients
  through means/conics/compact weights/lookup/opacities. It also verifies the
  current ID skeleton matches densified compact coefficients.
- `v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py` now
  records bounded direct-F32 versus compact K lookup timings. Saved local
  artifacts cover `128px/G2048/F32` for B4 and B16 with K=4/8/16.

The result changes the lookup branch status from "math/API only" to "timing
candidate, memory unproven." Lookup was faster in those bounded rows, but the
sampled MPS allocation was mixed and is not a true peak-memory counter. The main
promotion blockers remain real peak-memory evidence, trainer fixed-render
evidence, and W&B heldout-quality parity.

## Post-Audit Sampled-Memory Update

`v6_feature_lookup_experiment/benchmarks/benchmark_lookup_basis.py` now includes
a background sampler for `torch.mps.current_allocated_memory()` and
`driver_allocated_memory()` during the measured forward/backward window. This is
still not a Metal hardware capture, but it is stronger than only reading memory
after synchronized backward.

Saved artifacts:

- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_sampled_peak_128_b16_g2048_f32_k4_8_16.jsonl`
- `benchmark_outputs/fast_mac_feature_kernels/2026-05-07_lookup_basis_sampled_peak_128_b16_g8192_f32_k4_8_16.jsonl`

Read:

- At `128px/B16/G8192/F32`, lookup stayed faster for K=4/8/16 and lowered
  sampled current allocation versus direct F32.
- At `128px/B16/G2048/F32`, lookup was faster, but K=16 used more sampled
  current allocation than direct.
- This upgrades the lookup branch from "memory unproven" to "sampled-memory
  promising at the larger synthetic row." It still does not clear trainer
  integration, true peak-memory, or heldout-quality gates.

## Post-Audit Trainer Sampled-Memory Update

`src/benchmarks/trainer_phase_benchmark.py` now has opt-in sampled allocation
tracking:

```bash
--memory-sample-interval-ms 1.0
```

The default remains off, so older timing artifacts stay comparable. With the
sampler enabled, each measured iteration records start/end allocation counters,
sampled peak current allocation, sampled peak driver allocation, and sample
count. The row is still not a Metal hardware capture, but it is closer to the
actual train loop than the synthetic lookup probe.

Saved 256px fixed-render artifacts:

- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_fixedbin_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_accum_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_reduce_fixed_render_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_sampled_memory_seed0_warm1_iters2.json`

Read:

- Stable `v6_refined_features`: `673.9ms` total, `525.8ms` backward,
  sampled peak current `1412745984` bytes.
- `f32_gradcache`: `649.3ms` total, `502.3ms` backward, sampled peak current
  `1412745984` bytes.
- `f32_fixedbin`: `649.8ms` total, `500.6ms` backward, sampled peak current
  `1700988160` bytes.
- `f32_accum`: `670.2ms` total, `513.0ms` backward, sampled peak current
  `1412745984` bytes.
- `f32_reduce`: `667.4ms` total, `510.5ms` backward, sampled peak current
  `1636237568` bytes.

Updated interpretation: `f32_gradcache` is currently the cleanest trainer-path
timing/memory candidate in the bounded 256px row. Fixedbin remains a useful
host/binning experiment, but this trainer evidence weakens the idea that it is
a memory-pressure fix.

## Post-Audit Trainer Microbatch Probe

`src/benchmarks/trainer_phase_benchmark.py` now has benchmark-only fixed-render
microbatch knobs:

```bash
--fixed-render-temporal-chunk-size 8
--fixed-render-backward-mode chunked
```

The probe splits each fixed 16-frame train view into temporal chunks and
backprops each chunk immediately. It intentionally does not change the trainer
yet.

Saved 256px fixed-render artifacts:

- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_chunk8_chunked_backward_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_v6_refined_features_fixed_render_chunk4_chunked_backward_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_chunk8_chunked_backward_sampled_memory_seed0_warm1_iters2.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_f32_gradcache_fixed_render_chunk4_chunked_backward_sampled_memory_seed0_warm1_iters2.json`

Read:

- Stable full-target batched backward: `673.9ms`, sampled current `1412745984`.
- Stable chunk size 8 chunked backward: `770.9ms`, sampled current `567055872`.
- Stable chunk size 4 chunked backward: `910.1ms`, sampled current `365686016`.
- `f32_gradcache` full-target batched backward: `649.3ms`, sampled current
  `1412745984`.
- `f32_gradcache` chunk size 8 chunked backward: `1257.7ms`, sampled current
  `567056896`.
- `f32_gradcache` chunk size 4 chunked backward: `1528.1ms`, sampled current
  `365686016`.

Updated interpretation: temporal render/loss microbatching is a larger memory
lever than the current kernel forks. Chunk size 8 looks like the best first
tradeoff on the stable row: about `60%` lower sampled current allocation for an
`~14%` timing cost. The shared-background rerun made `f32_gradcache` chunked
backward much slower, so chunking and kernel choice should be benchmarked
together in the same session before promotion. Real trainer wiring still needs
a parity/quality smoke because it changes backward accumulation order and must
preserve shared train-background semantics.

## Post-Audit Chunked-Backward Parity Gate

`src/benchmarks/fixed_render_backward_mode_parity.py` now compares full-target
batched backward against temporal chunked backward on the same fixed render/loss
graph. It aggregates chunk gradients back into the full 16-frame layout before
comparing.

Saved 256px train artifacts:

- `benchmark_outputs/trainer_phase/multicam256_v6_refined_features_chunk8_vs_batched_backward_parity_seed0.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_gradcache_chunk8_vs_batched_backward_parity_seed0.json`

Read:

- Stable `v6_refined_features`: loss abs diff `7.45e-09`, max sequence grad
  diff `8.15e-10`, max colorize grad diff `4.05e-08`.
- `f32_gradcache`: loss abs diff `7.45e-09`, max sequence grad diff
  `8.44e-10`, max colorize grad diff `4.05e-08`.

Updated interpretation: chunk8 fixed-render backward is gradient-equivalent to
batched backward to MPS noise for both the stable path and the current batched
timing candidate. This still does not clear real multicam trainer wiring,
because camera-swap relpose, regularizers, optimizer order, and W&B heldout
validation are outside the fixed-render graph.

## Post-Audit Whole-Step Memory Smoke

`src/benchmarks/train_step_memory_benchmark.py` now times and samples memory
around the actual `trainer.step()` call. This is separate from fixed-render
phase isolation and exercises the active `camera_swap_mode=learned_residual`
path.

Saved 256px train-step artifacts:

- `benchmark_outputs/trainer_phase/multicam256_v6_refined_features_train_step_sampled_memory_seed0_iters1.json`
- `benchmark_outputs/trainer_phase/multicam256_f32_gradcache_train_step_sampled_memory_seed0_iters1.json`

Read:

- Stable `v6_refined_features`: `4822.1ms`, sampled current `2933897216`,
  sampled driver `3670294528`, loss `0.3316246`.
- `f32_gradcache`: `3952.1ms`, sampled current `2933896192`, sampled driver
  `3670294528`, loss `0.3316246`.

Updated interpretation: the real learned-residual step supports the full-batch
`f32_gradcache` timing direction, but memory is unchanged and much higher than
fixed-render because camera-swap renders multiple source/query pairs and keeps
relpose/cycle graph pieces live. This is a one-step smoke, not a warmed
throughput benchmark or promotion gate.
