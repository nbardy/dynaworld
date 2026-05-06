# Feature Kernel Goal Completion Audit

## Objective Restatement

The active objective was to iterate on copied Metal feature-splatting kernels,
benchmark larger batch/feature cases, drive raster speed down, write shared
notes about Metal shader performance and bottlenecks, read relevant Metal
documentation, and keep the machine safe by avoiding sloppy/high-risk shader
sweeps that could freeze the GPU.

Success criteria for this work chunk:

1. Stable kernels remain untouched; experiments live in copied forks.
2. Multiple forked variants exist and are selectable only by opt-in config or
   benchmark script.
3. Benchmarks cover larger batch and feature pressure rows, not only tiny smoke
   cases.
4. At least one fork measurably improves high-pressure F-channel timing.
5. Safety gates exist: local correctness tests, parity tests, safe benchmark
   runner rules, bounded tensor sizes, and no 4K/64K stress jumps.
6. Shared docs explain measured bottlenecks, Metal performance mechanisms, and
   official-doc implications.
7. Agent notes preserve the failed ideas, surprises, and promotion blockers.

## Prompt-To-Artifact Checklist

| Requirement | Concrete evidence | Audit status |
| --- | --- | --- |
| Fork kernels instead of mutating the working one | `git -C third_party/fast-mac-gsplat diff --stat -- variants/v5 variants/v5_features variants/v6_refined_features variants/v6_refined_features_f32_reduce` returned empty after the zero-bg fork. Root dispatch only adds opt-in `render.fast_mac.feature_variant = "v6_refined_features_f32_zero_bg"`. | Covered |
| New copied kernel versions | Fork dirs exist for `v6_refined_features_f32_reduce`, `f32_accum`, `f32_gradcache`, `f32_fixedbin`, `f32_zero_bg`, `f32_block4`, `f64_accum64`, `f32_stage`, and `v6_feature_lookup_experiment`. The newest copied fork is committed in submodule commit `1e45b4d`. | Covered |
| Benchmarks for bigger batches/features | `research_notes/fast_mac_feature_metal_performance.md` records rows for `B16/B32`, `F32/F64`, `256/512px`, frozen-color rows, trainer fixed-render rows, train-step sampled-memory rows, and lookup sampled-memory rows. | Covered |
| Speed driven down | Stable `512px/B16/G8192/F32` corrected-cap row: `1001.9ms`; `f32_reduce`: `449.2ms`; `f32_accum`: `388.1ms`. Later candidate rows include `f32_gradcache`, `f32_fixedbin`, and zero-bg bounded wins over `f32_reduce`. | Covered |
| F3 red flag investigated | Corrected-cap F3 feature forward matches RGB-class timing; stable feature F3 backward overhead was traced to generic feature-gradient handling, and `f32_reduce` recovers most of it. | Covered |
| Batch/memory pathology investigated | Notes distinguish Gaussian count from dense `[B,H,W,F]` surfaces; train-step sampled-memory artifacts show camera-swap pair count is a larger real-step memory lever than current kernel forks. | Covered |
| Safe benchmarking | Safe benchmark contract limits default size, requires warmup/timeouts, avoids overflow stress unless intentional, forbids parallel GPU benchmark launches, and records the largest local-safe pressure rows. | Covered |
| Correctness and parity gates | Fork-local feature/alpha checks pass for current candidates; fixed-render parity covers train/heldout forward and gradient parity for major forks; camera-swap parity exists with the documented MPS input-norm caveat. | Covered |
| Shared Metal shader performance docs | `research_notes/fast_mac_feature_metal_performance.md` has artifact map, measured tables, Metal feature coverage, bottleneck taxonomy, safe benchmark contract, profiling workflow, and next-fork decisions. | Covered |
| Read relevant Metal documentation | The shared doc links and maps official Apple docs for threads/threadgroups, SIMD width, pipeline threadgroup limits, threadgroup memory, compute passes, storage modes, argument buffers, heaps, feature-set limits, occupancy, memory bandwidth, shader validation, and workload captures. | Covered for compute-rasterizer-relevant Metal features |
| Preserve failed ideas and surprises | Loose notes and `agent_notes/key_learnings.md` record cap/fallback misdiagnosis, staging regression, block4 regression, F64 accumulator negative, fixedbin memory tradeoff, and the rejected implicit-zero output-init shortcut. | Covered |
| Commit order and clean state | Submodule committed first (`1e45b4d Add zero-background feature fork`), then root pointer/docs dispatch committed (`e3c8db6 Wire zero-background feature fork`). Worktrees were clean before this audit note/doc refresh. | Covered |

## Current Evidence Snapshot

Recent committed work:

- Submodule: `1e45b4d Add zero-background feature fork`
- Root: `e3c8db6 Wire zero-background feature fork`

Audit verification commands:

- Artifact existence check over `research_notes/fast_mac_feature_metal_performance.md`
  and this audit note returned no `MISSING` lines.
- Stable-kernel diff check over `variants/v5`, `variants/v5_features`,
  `variants/v6_refined_features`, and `variants/v6_refined_features_f32_reduce`
  returned empty output.
- Root and submodule `diff --check` passed.

Zero-background validation:

- `alpha_output_check.py`: tests A-F passed.
- `feature_contract_check.py`: shape/F3 parity/F3-F64 gradient/no-NaN checks
  passed.
- Parent-vs-fork MPS parity probe at `B4/G512/H64/W64/F32`: zero feature and
  alpha output diff for active off/on; max gradient drift `2.98e-08`.

Representative measured wins:

- Stable `512px/B16/G8192/F32`: `1001.9ms` corrected-cap total.
- `f32_reduce` same row: `449.2ms`.
- `f32_accum` same row: `388.1ms`.
- `f32_zero_bg` bounded row versus `f32_reduce`:
  - `256px` active off: `386.0ms -> 361.1ms`
  - `256px` active on: `472.0ms -> 454.6ms`
  - `512px` active off: `610.7ms -> 556.7ms`
  - `512px` active on: `813.4ms -> 761.9ms`

## What Is Not Claimed

- No experimental fork is promoted to default trainer config.
- No W&B heldout-quality parity run is claimed for the new renderer choices.
- No Xcode hardware-counter capture has been saved; local profile fields are
  logical raster stats, not Metal counter evidence.
- The feature lookup prototype is not trainer-integrated and is not a true
  sparse-ID kernel yet.

These are future promotion gates, not blockers for the completed objective as
stated. The objective asked for kernel iteration, benchmarking, speed work,
Metal notes, and safe execution; it did not ask to make a fork the production
default.

## Final Audit Conclusion

The active kernel-performance work chunk is complete. The repo now has copied
forks, bounded benchmarks, measured high-pressure speedups, safety rules,
correctness/parity gates, shared Metal performance docs, and durable agent
notes. Future work should start from promotion-quality evidence: Xcode
counter captures on finalist rows, W&B heldout-quality parity for any selected
fork, and trainer integration only where fixed-render and camera-swap parity
already pass.
