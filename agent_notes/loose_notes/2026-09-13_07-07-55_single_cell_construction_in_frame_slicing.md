# One cell construction preserves frame slices and reduces measured CPU work

## Scope and previous turn

The preceding user-facing turn recovered and explained existing source-overfit
results but changed neither authoritative state nor accepted evidence counts. It
was no new goal progress. This continuation revalidated the current worktree,
finished the prior verified fallback integration as root ce3bede, and targeted
measured duplicate cell construction in frame slicing. One lead, no subagents,
sequential CPU/Metal diagnostics, unchanged resource limits and offline W&B.
The broad overnight iteration remains active without a new token budget.

## Exact implementation change

STAR f3544aa changes only slice_projective_trace_cell_atlas_frames, reducing two
net source lines. The old code constructed a clipped frozen dataclass and then
constructed it again to remap primitive IDs. It now stores the original cell
with its clipped start/stop in a local tuple and performs both updates together.
No new renderer, coefficient law, native binary, retained atlas field or cache.

The two updates write disjoint dataclass fields. The first only writes start and
stop, so the source-ID set and sorted remapping are identical when computed from
the original cell. Thus replace(replace(cell, time_fields), id_fields) equals
replace(cell, time_fields, id_fields). The concrete frozen cell class has no
post-init behavior. Clipping casts, inequalities, cell order, depth intervals,
fallback flags/reasons and empty/invalid-slice behavior remain unchanged.
Tensor index_select operations and all optional fields are unchanged, preserving
the same live autograd connections. This is a constant-factor implementation
optimization, not new mathematical expressivity or a sublinear algorithm.

## CPU and actual Metal verification

The existing CPU/Metal suite passes 174 tests in 4.15 s, including empty slices,
reference whole-versus-chunked image/VJP agreement, live source updates and
world/trace gradient checks. No implementation-call-count test was added.
Terminal test session: 83448, exit zero.

The mechanical comparator extracts the exact old slice function from the retained
source AST, loads verified fitted-world atlases for F4/8/16/32, and compares every
atlas dataclass field. All 60 one-frame chunks and 12 wider/interior/empty windows
match exactly, including optional tensor values, dtype/device and metadata.
Existing regression tests provide live gradient coverage and positive fallback
fixtures; these retained learned atlases have zero fallback cells.

One warmup per route plus three alternating paired repeats at F32 give CPU
slicing medians 1.081519 -> 0.847015 s for all 32 frames
(1.277x). Raw old samples are [1.0815186249965336, 1.0842790000024252, 1.0606333330069901];
new samples are [0.847014750004746, 0.8712712499982445, 0.8407718750095228]. This timing excludes
rendering and is not presented as a GPU or optimizer speedup. Terminal session
33954 exits zero. Old source SHA: 84a19929b45691dab8f8ba7c1dc73e55c4de76744451df8eefff53306f3ba4bb.

The full learned-world Metal sweep (terminal session 88989, exit zero) retains
the same 800-update/2048-tube checkpoint, cam06, 96x128 targets, F4/8/16/32 over
one fixed physical interval, one-frame resident chunks, CPU target LRU8, one
warmup and three alternating replay/compiled timing repeats. Existing independent
payload/storage/memory/timing/slicing validators pass all four local rows
(validator session 11507, terminal exit zero).
All retained atlas bytes, topology, active sets, target/camera/world/time contracts
and acceptance thresholds equal the previous run. Maximum RGB error is
8.64267349e-07; maximum per-parameter normalized VJP error is
1.66195171e-06. F4 selected-time slice parity also passes.

Times are seconds. Evaluator+backward excludes target I/O, transfer and loss;
full cost includes compilation and all measured forward/backward phases, but
not an optimizer update or inter-segment cleanup. Before/after full sweeps are
separate processes; only the isolated CPU comparison above is interleaved in
one process. Small changes in unrelated phases are not causally attributed to
this edit. Medians of separate phases need not sum to the median same-trial sum.

| F | Evaluator+backward before | After | Full cost before | After |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.15830 | 0.14769 | 0.59165 | 0.58835 |
| 8 | 0.42988 | 0.39746 | 1.48668 | 1.43694 |
| 16 | 1.26491 | 1.22642 | 6.24826 | 6.22890 |
| 32 | 3.15545 | 2.96249 | 12.46966 | 12.17338 |

F32 compilation is 3.02231 -> 2.89246 s. Current CPU target
loading takes 6.27109 s; replay evaluator+backward is
0.52745 s.
The compiled evaluator+backward grows 20.059x for 8x more
frames. This does not establish sublinear total rasterization or a speed win over
replay. Local numerical rows remain 4/4; public counts remain 0/7 contexts and
0/21 lanes, publication eligibility false and BASELINES unchanged. No training
run occurred, so the successful old source fits and current shared-world
20.84/15.30-dB train/validation result remain unchanged.

## Resources and next evidence

The actual full sweep peaks at 2046279680 bytes (1.906 GiB) process-tree plus
launcher RSS, with 0 bytes new host swap and no resource trip. The CPU
comparison peaks at 579272704 bytes and tests at 922173440 bytes; both have zero
new swap. Limits remain 3-GiB process RSS, 2-GiB MPS allocator, 256-MiB new swap,
600-second wall time and the original host/disk/output limits. The host is the
real 24-GiB machine, not physical 8-GiB hardware. No application was killed,
online W&B upload made, or stopped shape-recording profiler retried.

Offline W&B: 80uag4t6. Native SHA remains
6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.
outputs/benchmarks/2026-09-13_single_cell_slice/ retains before/after source, raw paired
CPU samples, report/atlas identities, regression logs, resource receipts, all
Metal timing trials, offline W&B and the application of the existing verifier.

Next test overlap indexing for sequential frame slices: current slicing still
scans all C atlas cells for each of F frames, retaining O(FC) scan work before
materializing active cells and copying/remapping trace tables. The retained
pre-change F32 profile records 7,929,650 min and max calls each across 160
slices, versus roughly 6,352 retained cells per slice. An interval lookup could
reduce scans without changing the one-frame memory/data contract, but its build,
memory and invalidation costs must be measured, and source cell order preserved.
Do not claim that indexing already exists or that these finite curves prove an
asymptotic result. Keep target decoding separate in the cost ledger.
