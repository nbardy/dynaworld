# Reusing identical visibility queries cuts fitted-world compilation cost

## Scope and preceding evidence

The previous goal turn was progress: retained forward inputs fixed a reproduced
backward-state bug and improved actual F4/8/16/32 timings. This turn followed the
remaining measured cost with a bounded F32 Python profile, then changed one
compiler function and measured it. One lead, no subagents, sequential accelerator
jobs, offline logging and unchanged host/RSS/MPS/swap/disk limits. All diagnostics
retain a 600-second wall bound. The broad overnight goal stays active; no new
token budget was supplied. The failed Kineto/record_shapes lane was not retried.

## F32 Python profile

outputs/benchmarks/2026-09-13_f32_call_profile/ holds synchronized, non-overlapping
Python call profiles of atlas construction, frame slicing, forward and native
backward on the same fitted 2048-tube world. It evaluates only F32 over the fixed
32-frame interval, with one warmup and three timing repeats. No shape recording,
tensor capture, Kineto or native rebuild is involved. Session 64026 is terminal
exit zero. Existing independent validators pass, the retained atlas is byte-exact
to the prior F32 run, and all image/world-gradient gates pass. Offline W&B:
x4sqh1ka. Run-bound sources are preserved and were hash-verified before edits.

Across five compiler calls, profiled wall time is 51.17 s: continuous visibility
stratification is 32.23 s cumulative, fallback marking 11.14 s and support rebinning
7.67 s. There are 1,393,295 pair-root queries and 2,597,525 scalar depth-range
queries. Across 160 frame slices, wall time is 12.88 s, including 2,032,840
replacements costing 6.04 s cumulative. Forward wall is 9.63 s, including 4.93 s
packing and 3.87 s synchronization. Native backward wall is 1.94 s, including
1.81 s synchronization; the duplicate packing is gone.

These profile timings include profiler overhead and synchronization and must not
be promoted as ordinary throughput or independent GPU kernel timings. They
select visibility compilation as the next concrete bottleneck. Profile peak
tree-plus-launcher RSS is 1,924,923,392 bytes, with no new swap or guard trip.

## Implementation and equivalence

STAR commit c97e0c2 changes stratify_projective_trace_cell_atlas_visibility_events
only. Two dictionaries live for one invocation. Pair roots are keyed by ordered
trace IDs and the exact overlap start/stop indices. Depth ranges are keyed by
trace ID and exact start/stop indices. Repeated queries from different spatial
tiles reuse the result; distinct time bounds still compute their own result.
The source change adds three net lines. No native ABI, shader, loss, camera,
geometry, coefficient arithmetic, tolerance or serialized atlas layout changes.

The equivalence argument is direct: coefficients, sampled times, validity masks
and root epsilon are fixed for the invocation. Each key therefore determines
all arguments to the original pure computation. The first query executes that
same Torch subtraction/root solver or masked min/max operation; later queries
return the identical float tuple. Cell-specific root-boundary insertion remains
outside the cache. Midpoint ordering, source-ID ties and fallback marking are
unchanged. The dictionaries disappear after this compilation, so later geometry
or time updates cannot reuse old results. Extra temporary CPU storage is included
in the measured process-tree RSS; retained atlas storage remains unchanged.

A new CPU regression uses the same quadratic trace pair in three tile windows:
its -1 and +1 crossings belong to different spans. Joint compilation must equal
independent per-tile compilation, including depth ranges, and report zero sample
order mismatches. Repeating with changed geometry protects freshness between
calls. This tests the visible partition/order contract rather than cache call
counts. The full focused CPU/Metal suite passes 174 tests in 4.11 s, including
existing gradients, root ties, fallback, anisotropy and live-update cases.
Test session 27686 is terminal exit zero.

## Paired CPU comparison

A mechanical comparison extracts the exact old function from its retained AST
and executes old/new functions in the same process on identical inputs. Actual
coefficients and active intervals come from independently verified fitted-world
atlas files. Support cells are regenerated on CPU for this fixture; the fixture
is explicitly not claimed as byte-exact MPS pre-compilation state.

Every output cell field is exact at F4/8/16/32, including 48,753 output cells in
the F32 fixture. One warmup per implementation precedes three alternating paired
trials. F32 uncached samples are 4.97757/4.91626/5.03714 s; memoized samples are
1.44928/1.51588/1.47793 s. Medians are 4.97757 -> 1.47793 s, a 3.37x speedup in
this CPU visibility operation. No samples were removed. This mechanical check
is not a training or publication row. Session 62597 is terminal exit zero;
peak tree-plus-launcher RSS is 677,543,936 bytes with zero new swap.

## Full fixed-world Metal sweep

The actual replay/compiled sweep retains the same 800-update Coffee Martini
checkpoint, cam06, 96x128 targets, F={4,8,16,32} across the same physical interval,
one-frame resident chunks and an eight-frame CPU target LRU. One warmup and three
alternating replay/compiled timing repetitions are retained at each F. Session
66045 and validator session 12696 are terminal exit zero. All four image and
world-gradient rows pass; maximum RGB error is 8.64e-7 and maximum per-parameter
normalized gradient error 1.67e-6. F4 selected-time slicing passes. Every atlas
SHA/byte count, topology, active set, target/camera/world/time contract and
acceptance tolerance matches the previous run exactly.

Below, compilation and full cost are seconds. Full cost sums the measured
compilation, forward and backward segments, including target I/O and loss; it
excludes optimizer/inter-segment cleanup. Before and after full sweeps use
separate processes, so they are less isolated than the paired CPU operation.

| F | Compile before | Compile after | Full cost before | Full cost after |
| --- | ---: | ---: | ---: | ---: |
| 4 | 1.00824 | 0.60563 | 1.16003 | 0.74712 |
| 8 | 2.58243 | 1.39464 | 3.03137 | 1.81959 |
| 16 | 5.13101 | 2.57451 | 9.61079 | 7.03467 |
| 32 | 8.18933 | 4.36936 | 17.80326 | 13.83092 |

At F32, compilation falls 8.18933 -> 4.36936 s (46.64% less time), and full cost
falls 17.80326 -> 13.83092 s (22.31% less time). Evaluator+backward changes only
3.21791 -> 3.16827 s, consistent with a compiler-only change plus run variation.
Remaining separately summarized medians include 2.70592 s evaluator forward and
6.28072 s CPU target loading. A sum of separate medians need not equal the median
of same-trial sums. The existing validators check samples and phase telescoping.

Compilation grows 7.21x for 8x frames in this finite curve, but evaluator+backward
grows 21.84x. Neither is an asymptotic result: interval cell count grows from
1,853 to 48,753, and denser sampling reveals more active traces/interactions.
Do not turn the compilation ratio alone into a sublinear rasterization claim.
This is compiler engineering, not new gauge mathematics. Local numerical rows
remain 4/4; public counts remain 0/7 contexts and 0/21 lanes. Publication
eligibility is false, BASELINES is unchanged, and source-fit quality is unaffected.

## Resources, artifacts and next action

The full sweep peaks at 2,071,707,648 bytes (~1.93 GiB) process-tree plus launcher
RSS, with zero new swap and no guard trip. The focused test peaks at 902,955,008
bytes. All retain the 3-GiB RSS, 2-GiB MPS allocator and 256-MiB swap-growth limits
on the real 24-GiB host. The prior full sweep had ~1.74 GiB RSS; this change does
not claim lower peak memory. Caches add temporary CPU objects.

outputs/benchmarks/2026-09-13_visibility_memoization/ retains old/new source snapshots, exact cell comparisons, all trial
samples, CPU/Metal logs, native identity, complete atlases/rows, and the existing
independent validators' application. Offline W&B xklwsw9b contains the identical
report. Native SHA remains
6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.

Next address repeated scalar tensor reads in visibility fallback marking, a
measured unchanged compiler cost, using exact host values while retaining the
general spatial-depth path and metadata checks. Frame slicing is another measured
cost; avoid mixing those changes before an isolated comparison. Keep target
loading visible in the cost ledger. Preserve the old successful source fits and
the shared-world 20.84/15.30-dB quality control; no optimizer fit ran this turn.
Root integration owns the visibility regression, this note, scoped status edits
and the STAR gitlink. Unrelated benchmark, browser and paper work is preserved.
