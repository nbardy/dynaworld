# Exact host depth reuse reduces fallback compilation cost

## Scope and decision

The previous goal turn was progress: visibility query reuse preserved all
fitted-world atlas bytes and reduced actual compilation time. This turn targets
the next unchanged hotspot from the retained F32 Python profile: fallback marking
made 2,597,525 scalar depth-range queries across five compiler calls. One lead,
no subagents, sequential accelerator work, offline logging and unchanged resource
limits. Every diagnostic retains a 600-second bound. The broad overnight goal
remains active, without a new token budget. No shape-recording profiler was retried.

## Implementation and numerical contract

STAR commit dc067d1 changes only mark_projective_trace_cell_visibility_fallbacks,
adding three net lines. After the existing CPU polynomial evaluation, it converts
the depth/validity columns to host values once. The spatial-tile loops reuse those
values. Exactly absent/zero spatial-depth polynomials use (depth, depth); nonzero
spatial-depth polynomials still use the original tile-corner calculation and UV
visibility-event report. All metadata, tile-domain and time/index checks remain.

The original scalar path read float32 elements through Tensor indexing and item.
Tensor.tolist returns the same numeric values in Python floats; every finite
float32 value is exactly representable there. The polynomial evaluation and its
operation order are unchanged, and invalid coordinates/depths retain the original
zero validity and fallback behavior. Sort keys, depth epsilon, tie handling,
cell ownership, sample segmentation and fallback reasons are unchanged. The
nonzero spatial-depth branch receives the same dense tensor as before.

The host table is local temporary O(trace_count * sampled_time_count) storage;
it is not a cross-update cache or a new retained atlas field. This existing
compiler already materializes a dense CPU trace/sample tensor. No native ABI,
shader, gradient law, coefficient parameterization, data, camera or loss changes.
The original live coefficient graph remains intact: metadata was detached before
this change too. This is an implementation optimization, not new mathematics.

## Correctness and isolated CPU evidence

The existing focused CPU/Metal suite passes 174 tests in 4.05 s, including exact
root ties, actual fallback cases, spatial depth-plane crossings, live geometry
updates, alpha membership and world/trace VJPs. No extra implementation-count
test was added for this conversion. Test session 59916 is terminal exit zero.

The paired CPU comparison extracts the exact old function from its retained
source AST, loads independently verified F4/8/16/32 atlas tensors and topology,
and supplies identical inputs to both functions. Every cell field is identical
at all four frame counts. These learned-world atlases have zero marked fallback
cells; the independent regression suite supplies the positive fallback cases.
The measured operation is fallback detection/marking, not fallback image rendering.

After one explicit warmup per implementation, three alternating paired F32
trials give old 1.60126/1.60504/1.54938 s and new
0.237045/0.239215/0.246256 s. Medians are 1.60126 -> 0.239215 s, a 6.69x speedup
for CPU fallback marking only. No samples were dropped. Peak process-tree plus
launcher RSS is 619,200,512 bytes with no new swap or guard trip. Session 2969
is terminal exit zero. This mechanical check is not an optimizer/publication row.

## Same fitted-world Metal sweep

The full sweep retains the same 800-update, 2048-tube Coffee Martini checkpoint,
cam06, 96x128 targets and F={4,8,16,32} across the same physical interval. One-frame
resident chunks, the eight-frame CPU target LRU, one warmup and three alternating
replay/compiled timing repetitions per row are unchanged. Session 18729 and
validator session 13662 are terminal exit zero. Offline W&B: 0pv8f0a9.

All four image/world-gradient rows pass; maximum RGB error is 8.64e-7 and maximum
per-parameter normalized VJP error is 1.68e-6. Selected-time slicing passes at F4.
Retained atlas SHA and byte count, topology, active sets, target/camera/world/time
contracts and every acceptance threshold match the previous run exactly. Existing
independent payload/storage/memory/timing/slicing validators accept all rows.

Times below are seconds. Full cost includes compilation and the measured
forward/backward segments, including target I/O, transfer and loss construction;
it excludes optimizer/inter-segment cleanup. Before and after full sweeps are
separate processes, while the CPU operation comparison above is interleaved in
one process. Do not attribute tiny unrelated phase differences to this source edit.

| F | Compile before | Compile after | Full cost before | Full cost after |
| --- | ---: | ---: | ---: | ---: |
| 4 | 0.60563 | 0.43978 | 0.74712 | 0.59165 |
| 8 | 1.39464 | 1.06204 | 1.81959 | 1.48668 |
| 16 | 2.57451 | 1.85677 | 7.03467 | 6.24826 |
| 32 | 4.36936 | 3.02231 | 13.83092 | 12.46966 |

At F32, compilation falls 4.36936 -> 3.02231 s (30.83% less time); full cost falls
13.83092 -> 12.46966 s (9.84% less time). The ~1.35-s compilation saving agrees
with the isolated ~1.36-s CPU-operation saving. Evaluator+backward changes only
3.16827 -> 3.15545 s. Separate phase medians include 2.69909 s evaluator forward
and 6.26100 s CPU target loading; replay E+B remains 0.50783 s. A sum of separate
medians need not equal the median same-trial sum.

Compilation grows 6.87x for 8x frames, while evaluator+backward grows 19.93x.
This finite curve does not establish asymptotic sublinear rasterization. Local
numerical rows remain 4/4; public counts remain 0/7 contexts and 0/21 lanes.
Publication eligibility is false, BASELINES is unchanged, and no fitting-quality
claim changes. The old successful source fits and shared-world 20.84/15.30-dB
training/validation result remain separate evidence; no optimizer ran this turn.

## Resources, provenance and next action

Peak process-tree plus launcher RSS is 2,012,545,024 bytes (~1.87 GiB) for the
sweep and 903,036,928 bytes for the tests. All runs have zero new swap and no guard
trip. Limits remain 3-GiB RSS, 2-GiB MPS allocator, 256-MiB swap growth and the
existing host/disk/output requirements. These measurements use the real 24-GiB
host, not physical 8-GiB hardware. The extra host table is covered by RSS; its
memory growth is not hidden behind the unchanged serialized atlas byte count.

outputs/benchmarks/2026-09-13_fallback_host_values/ retains before/after source, CPU comparisons, all raw timing samples,
complete atlas artifacts and reports, source/native hashes, offline W&B backing,
regression logs and resource receipts. Native SHA remains
6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.

Next address the measured frame-slicing cost: it constructs each clipped cell,
then constructs it again to remap trace IDs. Combine those constructions while
preserving clipping, every optional field and live tensor gradients; require
exact old/new slice metadata and full fixed-world checks. Do not mix this with
native ordering or data-cache changes. CPU target loading must remain explicit
in the total cost. This turn also recompresses the September compiler lessons
in key_learnings; detailed chronology remains in the retained loose notes.
Root integration owns this note, scoped status edits and the STAR gitlink.
Unrelated benchmark/browser/paper work remains untouched.
