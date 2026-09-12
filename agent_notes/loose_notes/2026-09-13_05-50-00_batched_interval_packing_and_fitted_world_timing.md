# Batched host packing improves real compiled evaluation with unchanged buffers

## Continuation and scope

The prior goal turn was progress: the in-trial timing breakdown passed its
independent checks and exposed expensive host packing. This turn changes that
measured operation and retains actual before/after evidence. One lead, no
subagents, sequential accelerator/native work, unchanged host/RSS/MPS/swap/disk
limits and offline logging. The broad overnight goal remains active, with no
new token budget. Public acceptance and source-fit quality are separate lanes.

## Implementation and exactness

STAR commit 8720d4e changes only pack_projective_trace_tile_time_bins in the
projective trace module. Counts, ids, starts, stops and overflow are filled in
ordinary host lists, then converted to contiguous int32 tensors once per buffer
on the requested device. This removes per-entry Torch scalar assignment and
reduces the function by four lines. Interval collection, same-id union, gap
preservation, dictionary/entry ordering, truncation, sentinels and overflow
counts are unchanged. No native kernel or binary changes, geometry changes,
atlas caching, frame-residency expansion or tolerance relaxation occur.

The equality argument is elementwise: both constructors start with identical
zero/-1 values and execute the same indexed writes, in the same order, with the
same integer values. The new code changes the container used during construction,
not the resulting five buffers. True counts can exceed capacity; overflow remains
reported and only the first capacity entries are stored. Empty tiles retain their
zero counts/bounds and negative ids. Host lists add transient construction storage;
the measured resource caps below still pass. This is a local bounded result.

A same-process CPU comparison extracts the exact old function from its retained
source using AST, imports the current dataclasses, and supplies identical real
frame chunks from the prior fixed-world atlas artifacts. Every one of the five
buffers matches bit-for-bit for all 60 chunks across F4/8/16/32. Ordered buffer
hashes and source-atlas hashes are retained. The old source SHA is
077cc4ef3f30de57c3a8254e53775f6dfa26dff94226ed397f0c4c214bb03ca0.

After correctness comparisons and one explicit warmup per route, seven alternating
paired trials on the real first-frame cells give median scalar 31.6476 ms and
batched 7.7499 ms: 4.08x faster CPU packing. Raw trials include an elevated batched
23.79-ms sample; no samples were dropped. This isolates the construction change
more tightly than subtracting times from different processes. It does not claim
a 4.08x full-renderer speedup.

Twelve new CPU/MPS cases protect true counts on overflow, exact-capacity behavior,
empty tiles, padding and temporal gaps. The full focused actual CPU/Metal gate
passes 171 tests in 3.97 s, including image/VJP, live geometry updates, depth,
visibility, quadrature, alpha membership and existing producer correctness.
No new test merely counts Tensor calls. The historical scalar oracle warning is
unchanged. Exact CPU comparison session 69411 and Metal gate session 72741 both
returned terminal exit zero before the full sweep began.

## Fixed-world measured results

The guarded sweep uses the same 800-update, 2048-tube Coffee Martini world,
cam06, 96x128 images and F={4,8,16,32} over the same physical interval. One-frame
resident chunks and the eight-frame CPU target LRU are unchanged. Every row has
one warmup and three alternating paired timing repetitions. Session 35793 is
terminal exit zero. All four image/world-gradient gates pass; max RGB error is
8.64e-7 and maximum per-parameter normalized VJP error is 1.66e-6. The configured
F4 selected-time slicing gate passes. Retained atlas bytes are exactly identical
to the prior run at every F; target/camera/world/time contracts, acceptance
thresholds, active sets and topology also match.

E+B below is the median same-trial evaluator-forward plus backward sum, excluding
target I/O and loss construction. It includes Python, packing, dispatch and
replay projection. Full cost adds compilation, target I/O and loss as measured
in the actual timed segments, but excludes optimizer/inter-segment cleanup.
All times are seconds. Before/after full sweeps ran in separate processes;
replay also changes somewhat, especially at F4, so do not treat every difference
as a perfectly controlled kernel speedup. CPU paired results support attribution.

| F | Compiled E+B before | Compiled E+B after | Improvement | Replay E+B after | Full compiled cost after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.55552 | 0.24958 | 2.23x | 0.03529 | 1.20722 |
| 8 | 1.32533 | 0.57788 | 2.29x | 0.06609 | 3.06497 |
| 16 | 3.20834 | 1.70286 | 1.88x | 0.26047 | 9.83330 |
| 32 | 7.02099 | 4.05433 | 1.73x | 0.54482 | 18.46540 |

At F32, full compiled forward+backward falls 13.3250 -> 10.3541 s. Including
compilation gives 21.4163 -> 18.4654 s, about 1.16x faster (13.78% less time),
while evaluator-only forward+backward improves 1.73x. CPU decoding remains
6.2662 s, compilation 8.1308 s, evaluator forward 2.7975 s and backward 1.2591 s
as independently summarized medians. A sum of these separate medians need not
equal the median measured sum. Replay's F32 E+B remains ~0.545 s, so compiled
E+B is still about 7.44x slower.

Absolute compiled time improves at every F, while its F4->F32 E+B growth ratio
increases from 12.64x to 16.24x for 8x frames. This is not a regression in absolute
runtime: reducing lower-order overhead can expose worse remaining scaling.
Replay's observed ratio is 15.44x in this run; sampled time coverage and actual
interaction counts also vary with F. Four local samples establish no asymptotic
exponent, and these results do not support sublinear total rasterization.
Accepted local correctness rows remain 4/4. Public counts remain 0/7 contexts
and 0/21 lanes, publication_eligible stays false, and BASELINES stays unchanged.

## Provenance, resources and next evidence

outputs/benchmarks/2026-09-13_batched_packing/ retains before/after source, old-function extraction, all-buffer hashes,
paired CPU trials, CPU/Metal tests, complete sweep rows/atlases, phase timings,
independent validation and resource receipts. Existing storage/payload/memory/
slicing/timing validators accept all new rows; no duplicate scientific verifier
was introduced. Offline W&B: ny2ny6t4. The native binary remains SHA
6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.

Peak sampled process-tree plus launcher RSS: CPU comparison 562,085,888 bytes;
Metal gate 916,520,960; full sweep 2,064,187,392. All have zero new swap and no
guard trip, under the unchanged 3-GiB process and 2-GiB MPS limits. These are
24-GiB-host measurements, not proof on physical 8-GiB hardware. All run/validator
handles are terminal; the full-world Kineto/record_shapes lane stays stopped.

Source inspection locates remaining candidates: forward and backward each pack
buffers; coefficient accessors repeatedly run positive-definiteness checks with
MPS scalar readback; slicing visits the full cell list for each chunk; native
ordering repeatedly scans candidates per pixel. Basic input checks themselves
are shape/dtype/device checks, not reductions. None of those costs is isolated
by the packing comparison. Next use a bounded Python call profile on a retained
F4 evaluator trial to choose between bin reuse, duplicate validation and native
ordering from measured costs. Preserve all geometry/mutation checks and complete
buffer accounting; do not remove correctness checks merely because they cost time.

Root integration owns the twelve buffer regressions, this note, scoped status
edits and the STAR gitlink. The unrelated benchmark/streaming/browser/paper WIP
remains uncommitted. Earlier successful source fits and the 20.84/15.30-dB shared-
world fit remain valid quality evidence; this pass changes execution, not training.
