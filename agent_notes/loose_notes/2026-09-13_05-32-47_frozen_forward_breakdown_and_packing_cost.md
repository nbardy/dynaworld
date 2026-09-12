# In-trial measurements isolate decoding and expose remaining evaluator overhead

## Continuation and completed evidence

The previous turn was progress: the pending cutoff sweep became terminal and
its existing independent validators accepted all four local rows. This turn
first integrated that repair (STAR 01bd9e0, root bfb3715), then measured the
remaining timing confound on exactly the same fitted world. New instrumentation
is STAR 91ee6b0. One lead, no subagents; accelerator/native jobs were sequential;
all existing resource gates remained in force. No training or GPU build occurred
in this timing/profile pass. The overnight goal remains active and unbudgeted.

## Measurement change

Repeated paired trials now partition forward into evaluator_forward,
target_cpu_load, target_transfer and loss. Boundaries synchronize the device
and use telescoping perf_counter timestamps. Replay's evaluator segment includes
projection and rendering; compiled's includes slice/state setup and rendering,
with its setup and render intervals summed across intervening target loading.
The same trial's full forward total includes all four phases. Backward and
compilation remain separately measured. No phase is inferred by subtracting a
measurement from another process. Resident chunks stay at one frame and the
CPU target LRU remains eight frames. No target-cache expansion hides its cost.

Additional barriers can affect performance and their overhead is included. These
are instrumented wall timings, not kernel-only GPU timings. Evaluator timing
still includes Python, packing and dispatch. Keeping input order preserves the
old replay-render-before-load and compiled-load-before-render sequences. Historical
single-shot correctness timing has no invented phase breakdown.

The existing independent timing verifier accepts older reports without this
optional breakdown. If a new breakdown is present, it must have all routes and
phases, finite nonnegative samples of the right count, independently recomputed
summaries and phase sums equal to its same-trial forward total. Eight new cases
check valid data and missing phases/trials, negative/NaN times, wrong summaries,
wrong totals and a separate-process provenance claim. The whole focused file
passes 20 tests in 0.47 s. These tests protect timing interpretation rather than
counting implementation calls.

## Same-world F4/8/16/32 results

The actual guarded MPS sweep completed (exact session 73756, exit zero), with one
warmup and three alternating paired repetitions per row. The independent
storage, payload, memory, timing and configured F4 slicing checks pass. All rows
pass image/world-gradient gates at the unchanged tolerances. Retained atlas
bytes are identical to the cutoff-repair run at every F; checkpoint, source
world, camera, targets, physical interval, topology and acceptance limits match.
Accepted local correctness rows remain 4/4. Public evidence stays 0/7 contexts
and 0/21 lanes; publication_eligible is false and BASELINES stays unchanged.

All table values are seconds. E+B is a median of per-trial evaluator-forward
plus backward sums; C+E+B adds that trial's compilation. It excludes target I/O
and loss construction, but includes host evaluator work. A sum of separate
column medians need not equal the median of the measured sum.

| F | Compiled CPU target load | Compiled evaluator forward | Compiled E+B | Replay E+B | Compiled C+E+B |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.00045 | 0.29836 | 0.55552 | 0.04283 | 1.52892 |
| 8 | 0.00135 | 0.73347 | 1.32533 | 0.07528 | 3.81139 |
| 16 | 3.13615 | 1.70698 | 3.20834 | 0.25840 | 8.20049 |
| 32 | 6.25030 | 4.29667 | 7.02099 | 0.54500 | 15.12974 |

Replay CPU target-load medians are 0.00028/0.00052/3.13304/6.26205 s. At F32,
compiled transfer and loss construction are only 0.02032 and 0.01527 s; its
measured evaluator forward is 4.29667 s. Thus decoding explains the large shared
jump above the eight-frame cache, but not the renderer gap. Compiled E+B grows
12.64x for 8x more frames and replay E+B grows 12.73x. Compiled E+B is about
12.88x slower than replay at F32. Compilation-inclusive evaluator cost is 15.13 s.

These four local samples do not establish an asymptotic exponent. Even under
one fixed world/physical interval, denser selected times change which primitives
are active: the earlier retained dense trace-sample proxy grows 51,016 -> 519,505
(10.18x), while cells grow 1,853 -> 48,753. Do not explain all runtime growth by
F alone or claim that the coefficient table's fixed size implies constant work.
The timing result is negative for the proposed end-to-end sublinear speed claim.

## Bounded CPU packing diagnosis

After the Metal sweep was terminal, a guarded CPU cProfile probe used the actual
F4 frame-zero cells, clipped and remapped exactly as the frame slicer: 570 cells,
561 traces, 10,488 packed active entries, maximum 78 per tile, zero overflow.
All five returned buffers match exactly across repetitions and the profiled call.
Three unprofiled packing calls take 55.98/32.68/32.06 ms. The profiled call takes
41.28 ms (43,688 calls); pack_projective_trace_tile_time_bins accounts for 41.03 ms
cumulative and 39.27 ms self time. Sorted interval calls account for only 0.63 ms.
The profile does not separate scalar Tensor indexing inside that self time, so
it does not by itself prove that every millisecond is scalar-write overhead.

Source inspection supplies the concrete next candidate: the packer writes three
individual CPU tensor elements per entry plus count/overflow values per tile.
The autograd bridge calls packing in forward and again in backward. Native
selection also scans candidates repeatedly per pixel, and frame slicing scans
all cells per chunk; their relative costs are not isolated by this profile.
No packer, slicing algorithm or native kernel was changed in this pass.

Next replace scalar buffer filling with batched CPU tensor construction, keeping
exact interval union, entry ordering, padding sentinels, count/overflow behavior
and all five returned buffers. Measure it against these retained real cells and
rerun the same fitted-world image/VJP/timing curve. Preserve forward/backward
semantics and resource caps; do not cache stale geometry or enlarge frame
residency. The full-world Kineto/record_shapes lane remains stopped.

## Resources, provenance and integration

outputs/benchmarks/2026-09-13_forward_breakdown/ retains rows, atlases, original/after source snapshots, native identity,
phase samples/summaries, independent validation, launch scripts, logs, receipts,
packing buffers' hashes and packing_cpu.prof. Offline W&B is hg03ue3e. Native
SHA remains 6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.
Peak tree/launcher RSS: sweep 2,081,374,208 bytes; CPU packing 271,597,568 bytes.
Both have zero new swap and no guard trip. Exact sweep/profile/validation handles
are terminal. This was on the 24-GiB host; physical 8-GiB operation is not claimed.

Root integration owns only the verifier extension, eight tests, this note,
scoped status edits and the STAR gitlink. Pre-existing streaming/runner/browser/
paper WIP remains unstaged, and the full measured benchmark source is retained
in after/. A nonessential plot attempt found matplotlib absent in both available
Python runtimes; no packages were installed, and the unused plotting script was
removed. The numerical table above is the retained presentation.
