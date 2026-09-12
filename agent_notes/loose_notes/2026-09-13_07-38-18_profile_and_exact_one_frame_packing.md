# Refreshed F32 profile and exact one-frame packing shortcut

## Scope and previous progress

The previous turn committed an ordered interval sweep (root 7884e9d / STAR
bb6c6f4), with exact cells and measured F32 improvement but mixed small-frame
timing. This turn refreshes the bounded profile, then fixes its demonstrated
packing cost. One lead, no subagents, sequential accelerator jobs, offline W&B,
unchanged resource guards and no new budget. The overnight goal remains active.

## Bounded profile before this packing change

outputs/benchmarks/2026-09-13_f32_post_slice_profile/ retains a synchronized
Python cProfile run on the same learned 2048-tube world and full F32 interval.
It profiles actual next(iterator) calls, not generator construction, so the first
slice includes event-index setup. This is not the stopped Kineto/record_shapes
mode. It passed the same-world/image/VJP/atlas gates, existing independent
validators and exact call-coverage checks: five compiles and 160 slices, forwards
and native backwards each. Session 55419 exits zero; its verifier completed with
exit zero. Native bytes and atlas are unchanged. Offline W&B 6y9p3whe.

These are accumulated instrumented times across those calls, not ordinary
benchmark wall times or speedup measurements:

| Profiled operation | Accumulated seconds | Relevant detail |
| --- | ---: | --- |
| Compiler | 21.242 | Visibility-event function 10.587 cumulative / 6.884 self |
| Streamed slices | 6.840 | 1,016,660 dataclass replacements take 2.644 cumulative |
| Forward wrapper | 9.369 | Packing 4.886 cumulative / 3.436 self |
| Forward GPU synchronization | 3.673 | Waiting is distinct from CPU packing |
| Native backward wrapper | 1.916 | 1.782 in synchronization |
| Quadratic interval bounds | 2.406 | 158,370 calls; 2.334 self, inside compilation |

Profile peak process-tree plus launcher RSS is 1,902,313,472 bytes, zero new
swap. Source snapshots were verified and archived before the following edit.
Do not rerun the historical current-source hash check against later code; its
after snapshots preserve the exact profile source. The profile changes the next
action: optimize general interval bookkeeping used on single-frame chunks.

## Implementation and exactness

STAR 44c8265 changes pack_projective_trace_tile_time_bins only. For one frame,
valid integer nonempty cell intervals are necessarily [0,1). Unioning repeated
instances of that interval for a trace yields [0,1) again. The fast branch thus
collects distinct trace IDs in first-seen order and emits one unit interval for
each. Python dictionary updates preserve an existing key's position. This is
the same trace ordering as the original insertion-ordered interval dictionary.

Fallback, image-tile, time-range and ID/depth-length validation occur before the
shortcut. The branch explicitly requires start=0 and stop=1. Counts include all
unique entries; capacity truncation, overflow flags and empty-slot values use
the unchanged output writer. General multi-frame sorting/union remains intact,
including touching intervals, gaps and distinct chart IDs. The shared unit
interval list is local and is never mutated; it is not a cross-update cache.

All five tensor constructors and their dtypes/devices remain unchanged. There
is no new native ABI, shader, Gaussian, camera, precision, opacity or gradient
law. Exact buffers imply the native forward/backward receives the same bins.
This is an algebraic special case of interval union, not new gauge theory or a
new asymptotic rasterization claim.

## Regression and CPU comparison

The focused real CPU/Metal suite passes 183 tests in 4.02 s (session 51899,
terminal exit zero). Two added cases protect duplicate IDs within/across cells,
first-seen order, overflow at small capacity, empty tiles and every packed buffer.
Existing tests cover true multi-frame interval gaps/union and Metal image/VJPs.

The CPU comparator extracts the exact old packing function from its retained
source, loads verified fitted-world atlases, and compares every field of the
five-buffer result for all 60 one-frame chunks at F4/8/16/32. It also checks
each full atlas with tile_t=1 and tile_t=F, preserving the general route.
All values, tensor types and devices match. Positive overflow is covered by
the regression fixtures; this is not a claim that arbitrary full-atlas packing
fits a fixed native capacity.

One warmup per route and three alternating paired F32 trials, including tensor
construction for all 32 chunks, give CPU medians
0.542433 -> 0.207573 s (2.613x).
Old samples: [0.5424331249960233, 0.542402666003909, 0.6299951250111917].
New samples: [0.20757329098705668, 0.20573112499550916, 0.21136641700286418].
Slicing is excluded from this isolated timing. Session 65816 is terminal exit
zero. Old source SHA: 031dee91ed8cc63268e5ed9ba746a65878f768c33f0af4c66c5022e7acb8b1f9.

## Full fixed-world Metal result

Session 33332 and the existing independent verifier application (session 6482)
are terminal exit zero. The same learned world, cam06, 96x128 targets, fixed
physical interval, F4/8/16/32, one-frame resident chunks, CPU target LRU8, one
warmup and three alternating replay/compiled repetitions are retained.
All four local rows pass; every retained atlas byte, topology and active set is
exact. Maximum RGB error is 8.64267349e-07; maximum normalized per-parameter VJP
error is 1.6580091e-06. F4 selected-time parity also passes.

Times below are seconds. Evaluator+backward excludes target I/O/transfer/loss.
Full cost includes compilation and measured forward/backward phases, but not
an optimizer update or inter-segment cleanup. Before/after full sweeps run in
separate processes; the CPU comparison above is interleaved within one process.
Do not causally attribute small changes in unrelated phases to this edit.

| F | Evaluator+backward before | After | Full cost before | After |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.14256 | 0.14254 | 0.60367 | 0.57204 |
| 8 | 0.40743 | 0.30461 | 1.45552 | 1.36543 |
| 16 | 1.18805 | 0.94451 | 6.22717 | 5.84786 |
| 32 | 2.73561 | 2.33777 | 12.07251 | 11.52976 |

F32 evaluator+backward falls 14.54%; full measured cost falls
4.50%. F4 evaluator+backward is essentially unchanged. F32 compilation is
2.93594 -> 2.88718 s and current CPU loading is 6.25135 s. Replay evaluator+
backward remains 0.52666 s. Separate medians need not add to a same-trial sum's median.
Compiled evaluator+backward still grows 16.401x for 8x frames and remains
slower than replay. Local rows stay 4/4; public counts stay 0/7 contexts and
0/21 lanes, publication eligibility false and BASELINES unchanged. No optimizer
fit ran, so source-fit positives and shared-world 20.84/15.30-dB quality are unchanged.

## Resources, integration and next action

Full-run peak process-tree plus launcher RSS is 2047033344 bytes (1.906 GiB),
with zero new swap and no guard trip. CPU comparison and regression peaks are
718,553,088 and 907,100,160 bytes, both with zero new swap. Limits remain 3-GiB
RSS, 2-GiB MPS allocator, 256-MiB new swap, 600-second wall time and the original
host/disk/output gates on the physical 24-GiB host. No applications were killed,
online uploads or remote compute used, or shape-recording profile retried.

Offline W&B p9hf3vvy; native SHA unchanged:
6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.
outputs/benchmarks/2026-09-13_single_frame_packing/ retains exact before/after sources, all
raw CPU/Metal samples, report/atlas identities, resource receipts and verifier
application. Root integration owns the two test cases, this note, scoped status
and learning changes, and the STAR gitlink. Existing unrelated WIP is preserved.

Next target the unchanged compiler's measured quadratic-bound cost. The helper
currently evaluates endpoints using scalar float32 Tensor arithmetic and tests
the vertex using Python floats. Simply replacing its operations with Python
float math changes rounding and can move support boundaries. Test batched
endpoint/vertex evaluation with exactly preserved precision, operation order,
boundary inclusion and record order before using it in support rebinning.
The refreshed profile supplies the baseline; no second profile is needed merely
because the independent packing path changed. Keep visibility correctness,
resource limits and the full measured cost explicit.
