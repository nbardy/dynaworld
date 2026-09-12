# Interval events remove full-cell scans from streamed frame slicing

## Scope and starting evidence

The previous goal turn was progress: root aa9415a / STAR f3544aa preserved exact
frame slices and reduced F32 evaluator+backward 3.16 -> 2.96 s. The remaining
full-cell scan per frame was demonstrated in source and in the retained bounded
Python profile. This turn builds and measures a transient overlap sweep rather
than caching across optimizer updates. One lead, no subagents, sequential jobs,
offline W&B and unchanged resource limits. The overnight goal remains active.

## Implementation and correctness argument

STAR bb6c6f4 adds iter_projective_trace_cell_atlas_frame_slices and exports it.
The helper snapshots the cell list when first advanced, sorts positive-length
intervals by start, and maintains active original cell indices with a stop heap.
For each half-open chunk [a,b), it removes cells ending at or before a and admits
previously unseen cells starting before b whose end is after a. Every admitted
cell therefore satisfies start < b and stop > a, exactly the original overlap
predicate after its integer conversions. Empty intervals remain absent.

Sorting active indices restores the original source-cell order. The unchanged
scalar slicer then clips times, compacts/remaps trace IDs, and selects all live
required/optional tensors. It receives the same overlapping cells in the same
order, so depth intervals, fallback metadata and tensor gradients are preserved.
The sweep retains a topology snapshot; a fresh iterator sees subsequent cell
edits. It never stores a global cache or reuses topology across optimizer steps.

The existing frozen-world correctness and timing routes now create a fresh lazy
iterator for each atlas. next(iterator) runs inside the timed forward section,
including all initial sorting and heap setup on the first chunk. The correctness
route's memory sampler also covers that work. Iterators are deleted after each
route/trial. No future frame image/tensor is eagerly materialized, and one-frame
resident chunks plus the eight-frame CPU target LRU remain unchanged.
The original random-access slicer and independent selected-time parity route
remain available. No Gaussian/camera/opacity/loss/native ABI changes occurred.

For C cells, K chunks and A_k overlapping cells in chunk k, the old candidate
scan costs O(K*C). The sweep uses O(C log C) event work plus
O(sum_k A_k log A_k) to restore source order, in addition to the existing active
cell construction, ID remapping and tensor selection. Auxiliary metadata is
O(C + max A_k); it does not duplicate references over every cell's full frame
span. This can lose when K is small or most cells span every chunk. It does not
make dense image output sublinear, nor guarantee a better overall exponent as
visibility-cell count grows. It is ordinary interval indexing, not new gauge math.

## Regression and paired CPU evidence

The actual CPU/Metal gate passes 181 tests in 4.07 s (session 63396, exit zero).
The existing whole-versus-chunked reference image/VJP test now exercises the
sweep with chunk sizes 1/2/3/7. A new behavioral test covers unsorted starts,
negative starts, exact end boundaries, empty intervals, temporal gaps, partial
last chunks, fallback metadata, preserved source order, and snapshot-versus-fresh
iterator behavior after a list edit. It runs at chunk sizes 1/2/3/12. No test
asserts private call counts or a chosen indexing implementation.

The CPU comparator loads independently verified F4/8/16/32 retained atlases and
compares every field against the exact pre-change slicer for chunk sizes
1/2/3/F+1, including required and optional tensors, type/device and metadata.
All fields match. The one-frame cases cover all 60 real chunks; the larger
windows also cover boundary crossings and incomplete final chunks. The saved
learned atlases have zero fallback cells; positive fallback fixtures remain in
the regression suite.

For all 32 F32 slices, including a newly constructed index on every trial,
one warmup per route and three alternating paired repeats give medians
0.863604 -> 0.678613 s (1.273x).
Old samples: [0.8636042500002077, 0.8638647920015501, 0.8508149580011377].
New samples: [0.6887707079877146, 0.6786129160027485, 0.6436044580041198].
Session 4063 is terminal exit zero. This is CPU slicing evidence, not a GPU
kernel or optimizer speedup. Before source SHA is
c2d0707ce783d9f399ec1a97b5eaf75018ddde22db10ff73bc661c91a41c2e71.
Both retained pre-change renderer and benchmark snapshots match the immediately
preceding run-bound hashes.

## Complete fitted-world Metal comparison

The same 800-update/2048-tube Coffee Martini world is evaluated at F4/8/16/32
from cam06, 96x128 targets and one fixed physical interval. The same resource,
image/VJP/fallback, sample-time, loss, warmup and repeat contracts apply.
The actual sweep (session 44714) and existing independent verifier application
(session 89152) both exit zero. All four local correctness rows pass; atlas
bytes, topology and active sets are exact versus the preceding run. Maximum RGB
error is 8.64267349e-07; maximum normalized per-parameter VJP error is
1.63470185e-06. F4 selected-time slicing also passes.

Times are seconds. Evaluator+backward excludes target I/O, transfer and loss.
Full cost includes compilation and measured forward/backward phases; it does
not include an optimizer update or inter-segment cleanup. These before/after
full sweeps are separate processes. Only the isolated CPU comparison above is
paired in the same process, so small unrelated phase changes are not attributed
to the sweep. A sum of separate medians need not equal a same-trial sum's median.

| F | Evaluator+backward before | After | Full cost before | After |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.14769 | 0.14256 | 0.58835 | 0.60367 |
| 8 | 0.39746 | 0.40743 | 1.43694 | 1.45552 |
| 16 | 1.22642 | 1.18805 | 6.22890 | 6.22717 |
| 32 | 2.96249 | 2.73561 | 12.17338 | 12.07251 |

F32 evaluator+backward falls 7.66%, while full measured cost falls only
0.83%. The smaller-frame results are mixed: F8 evaluator+backward rises,
and F4/F8 full cost rises. Do not claim a universal or across-the-board speedup.
F32 compilation is 2.89246 -> 2.93594 s and CPU loading is 6.39052 s.
Current replay evaluator+backward is
0.52235 s; compiled remains slower.
Evaluator+backward grows 19.189x for 8x frames, so the total
sublinear claim remains unsupported. Local rows remain 4/4; public counts remain
0/7 contexts and 0/21 lanes, publication eligibility false and BASELINES unchanged.
No optimizer fitting occurred; source-fit positives and the shared-world
20.84/15.30-dB train/validation result are unchanged.

## Resources, integration and next evidence

Full-run peak process-tree plus launcher RSS is 2012725248 bytes (1.874 GiB),
with zero new swap and no guard trip. Regression and CPU comparison peaks are
909361152 and 613203968 bytes, both with zero new swap. Limits remain 3-GiB RSS,
2-GiB MPS allocator, 256-MiB new swap, 600-second wall time and the original
host/disk/output gates. Metadata memory is covered by host RSS; it is transient
per-evaluation work, not uncounted persistent serialized atlas state. Native SHA
remains 6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.
The host is physically 24 GiB. No application was killed, online upload made,
remote compute used, or stopped shape-recording profiler retried.

Offline W&B: 94cxmgv1. outputs/benchmarks/2026-09-13_interval_slice_sweep/ retains all
source snapshots, CPU samples, complete atlas/report identities, raw Metal
samples, guards, logs, and the existing verifier application. The benchmark had
large unrelated WIP before this turn. The staged HEAD-based patch isolates only
the import and two iteration sites plus their cleanup; retained before/after
source hashes bind the actual tested worktree. Unrelated benchmark, browser and
paper work remains unstaged. Root integration owns the test change, this note,
scoped status/learning edits and the STAR gitlink.

Next refresh the bounded F32 Python call profile to rank the remaining active
cell construction/ID remapping, native buffer packing and GPU costs after both
slicing changes. Do not guess that the unchanged old profile proportions still
hold, or hide the separate target-loading cost. Keep the resource-failed
Kineto/record_shapes lane stopped. The current new implementation is a measured
local optimization; it does not resolve broad quality or publication scaling.
