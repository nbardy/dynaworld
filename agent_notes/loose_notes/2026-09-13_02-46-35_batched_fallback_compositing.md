# Batched fallback preserves the saved world and reduces warm rendering cost

The intervening source-fit clarification revisited existing evidence; it did
not advance accepted paper counts. This turn resumed the pending measured
fallback optimization. The old CPU process was confirmed terminal: its log
contained 92 passed/37 skipped, and a live process inventory showed no active
accelerator job. The broad overnight goal remains active.

STAR commit `68fa855` batches a selected fallback tile's fixed depth order.
This improves a demonstrated implementation bottleneck without changing the
world, compiler, atlas membership, thresholds, or alpha law. It is not new
gauge mathematics or an additional training/convergence result.

## What changes and why it preserves compositing

Previously, scalar sorting and validity checks repeatedly read GPU values,
and each contributing trace built a separate sequence of GPU operations.
The forward profile counted 9,525 Tensor.item calls, consuming 1.5662 s.
Depth and validity now copy their exact evaluated values to the CPU once.
Centered float32 source depth and source-primitive-id tie ordering are retained.

For an already sorted tile, let a_i(p) be the clamped/cutoff opacity and

    T_i(p) = product over j<i of (1-a_j(p)).
    RGB(p) = sum_i A_i*T_i(p)*a_i(p)*c_i.

The helper evaluates all tile alphas together and obtains T with an exclusive
cumprod. A_0=1; for later i, A_i=1 exactly when max_p T_i(p) exceeds the
transmittance cutoff (or all A_i=1 when the cutoff is disabled). Because
0<=a<=1 makes T nonincreasing, this reproduces the original tile-wide stop.
The first valid trace still contributes even for cutoff>=1. It does not
replace the tile-wide rule with independent per-pixel early termination.
The discrete stop mask is detached, matching the old branch's local gradient
semantics. A weight/color matrix product avoids a K*Htile*Wtile*C intermediate.
Other tile-local temporaries are O(K*Htile*Wtile).

Only the sparse fallback path with spatially constant depth order uses this
helper. Per-pixel spatial-depth sorting keeps its existing reference path.
All contributors from flagged and unflagged cells are still included in each
selected fallback tile. Full scalar reference rendering remains available as
the compositing comparison. Finite-precision reductions can differ slightly;
the original frozen image/world-VJP tolerances are unchanged.

## Repeated local Metal result

Same 256-tube saved world, four selected frames, cam06, 96x128, capacity128,
two-frame device chunks, peak-splat alpha, one warmup and three synchronized
alternating replay/compiled trials. The before implementation is STAR ccf6c1c.
These are consecutive process-level controls, not interleaved before/after
implementations or a clean-source publication sweep.

| Median seconds | Before | Batched fallback | Before/after |
| --- | ---: | ---: | ---: |
| Compilation | 0.94148 | 0.96687 | 0.97x |
| Compiled forward | 2.14532 | 0.17836 | 12.03x |
| Compiled backward | 1.81730 | 0.81759 | 2.22x |
| Compiled forward + backward | 3.96569 | 1.01800 | 3.90x |
| Compilation + forward + backward | 4.94081 | 2.00062 | 2.47x |
| Replay forward + backward | 0.04279 | 0.05689 | 0.75x |

The three compiled forward/backward sums are 1.01800, 1.03406, 0.99047 s;
compile-inclusive sums are 1.98487, 2.00062, 2.01680 s. Sum medians need not
equal sums of individual medians. Timing includes target transfer and world
projection, but excludes optimizer work, parity checks and inter-segment
cleanup. The cold correctness pass is still expensive (1.85 s forward,
2.52 s backward); the table's gain is explicitly warm. Replay variation is
another reason not to treat the ratios as hardware-independent constants.
Even the warm compiled path remains ~18x slower than replay here, or ~35x
including compilation. No speed-over-replay or sublinear scaling claim follows.

Offline W&B `0uw9h9ss` retains config, tags, report and timing distributions;
the preceding run is `i6pm2a9p`. No online sync was attempted. Mechanical
tests, slicing and profiles omit W&B; the repeated performance run logs offline.

## Numerical and storage evidence

F4 passes all eight local checks. Max RGB error is 6.8545341e-7; global world
VJP error is 8.0620000e-7 and maximum parameter-group error 7.7267846e-7.
All seven world parameter groups retain nonzero, matching gradient coverage.
F3 selected times [0,2,3] also pass: global VJP 1.0336135e-6, maximum parameter
error 1.1635623e-6, max RGB 6.8545341e-7. Its chunk-vs-single-frame slicing
comparison has exact RGB equality and global/max-parameter VJP errors
4.6477312e-7/5.3669065e-7. The unchanged numerical thresholds are 1e-5.

Existing independent timing, retained-storage and selected-time-slice validators
pass. Both F4 and F3 serialized atlas files are byte-identical to their previous
counterparts, proving unchanged tensors and topology. F4 is 1,038,473 bytes,
SHA `4476bfbce89f8748874a6decdbd86be3008b77f622040fb6f35f7d468cad89f5`.
Fallback is still 92/720=12.7778% for F4 and 17.0370% for F3. Checkpoint,
camera/target/config hashes, selected times, acceptance thresholds and all
common bound source hashes match. The world file SHA remains
`8ca44d076cf17e6aed274f2986a5cb033a7220fa2a1c14a41ca1e589345949f9`.

## Tests, resources, and remaining profile

- CPU depth/projective-correctness/producer/visibility suite: 92 passed,
  37 MPS skips; skipped checks were not counted as passes.
- Guarded Metal-enabled depth/fallback suite: 31 passed. New cases compare
  scalar and batched images/all five atlas-parameter VJPs with anisotropic
  precision, temporal opacity, saturated alpha, and tile-wide early stops.
  Existing source-depth rounding, source-id tie, contributor and spatial
  sorting regressions remain passing.
- All guarded runs completed without new swap or guard trips. Maximum sampled
  process-tree/launcher RSS was 1,822,441,472 bytes. F4 correctness-route
  current allocator peak fell 13,329,920 -> 10,646,784 bytes; sampled driver
  peak was 38,354,944 bytes. The 3-GiB RSS/2-GiB MPS and host/disk/600s limits
  were unchanged. These are measurements on the 24-GiB host, not an 8-GiB
  hardware certification.

The new instrumented forward has 13 Tensor.item calls (0.0109 s), and
_live_depth_key cost falls 0.6427 -> 0.000804 s. The helper still launches
460 index_select calls over 92 tiles; instrumented index_select time is
0.1296 s and tensor construction 0.0736 s. Python profiling attributes most
backward time to the autograd engine and synchronization; it does not identify
the responsible GPU operations. Profiles include first-use/profiler overhead;
use the separate repeated table for performance claims.

Artifacts, before/after source snapshots, source/resource receipts, profiles,
raw reports, and a replayable summary using existing validators are under
`outputs/benchmarks/2026-09-13_fallback_compositing/`. The checked-in benchmark
config remains `src/train_configs/frozen_fallback_cost_20260913.jsonc`.
No native rebuild, concurrent accelerator job, application termination,
subagent, or broad refactor occurred.

## Next action

The warm fallback backward (~0.82 s) and compilation (~0.97 s) remain the
largest measured costs. Inspect operation-level backward/gather costs before
another renderer change; a Python call-stack profile cannot assign them.
Preserve tile ordering, contributor coverage, saturated-opacity gradients and
the frozen tolerances. Dense image output still has an unavoidable cost
proportional to output pixels; this local reduction proves no temporal exponent.
Keep source-fit quality and shared-world training as separate measured controls.
Public paper counts remain 0/7 contexts and 0/21 lanes; BASELINES is unchanged.
