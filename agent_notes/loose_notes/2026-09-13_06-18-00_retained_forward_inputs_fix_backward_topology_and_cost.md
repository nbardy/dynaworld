# Retaining forward inputs fixes topology-sensitive backward and removes duplicate packing

## Continuation and scope

The preceding user-facing turn recovered and clarified historical source-fit
results but added no new experimental evidence. This continuation was therefore
revalidated as no progress toward the active experiment goal, and resumed the
pending measured backward repair. The worktree already contained two source edits
and two failing-before regression cases; those edits were inspected before use.
One lead, no subagents, sequential accelerator work, offline W&B, unchanged
resource gates and a 600-second wall limit per diagnostic. The broad overnight
goal remains active. No new token budget was supplied.

## Profile and reproduced defect

The retained bounded Python cProfile from
outputs/benchmarks/2026-09-13_evaluator_call_profile/ evaluates the same fitted
world at F4. Its source snapshots match every run-bound hash. It completed with
existing image/gradient/storage/slicing checks passing, exact atlas bytes,
peak tree-plus-launcher RSS 1,153,646,592 bytes and zero new swap. Offline W&B:
bjxzya7o. This is distinct from the stopped Kineto/record_shapes resource-failure
lane, which was not retried.

Across 28 profiled native-backward calls, total synchronized wrapper wall time
is 0.8561 s. Repacking bins accounts for 0.5151 s cumulative and final device
synchronization for 0.2426 s. Forward packs 32 times for 0.7462 s cumulative.
The separate slice profile makes 43,648 dataclasses.replace calls across 32
slice calls, costing 0.1222 s cumulative. cProfile changes Python overhead;
these are diagnostic attributions, not uninstrumented speed measurements or
independent per-kernel GPU timings.

Code inspection found that the old autograd context retained a mutable cell
list via static_atlas and rebuilt bins in backward. If a caller prepares a new
topology before backpropagating an earlier render, backward uses the new cell
membership with the old tensor values. Two actual Metal regressions render A,
mutate atlas.cells in place, render B, and differentiate A + w*B for w=0 and
w=0.35 under a colored cotangent. Both forward images match independent Torch
reference images, but the old coefficient gradients fail: 12/18 values differ,
maximum absolute error 3.58618. This demonstrates a valid multi-forward/autograd
failure; it is not evidence that this pattern caused earlier training PSNRs.

The retained failing-before command asserts pytest's TESTS_FAILED exit, so its
launcher correctly exits zero while the log records two failed tests. Source
snapshots, the new test bytes, and its no-trip resource receipt are checked.

## Repair and derivative contract

STAR commit 1aedaf4 extracts the existing validated native-input preparation
from the public plain-cell interval forward wrapper. The autograd forward saves
all fourteen exact native tensors: coefficients, times, opacity, temporal
coefficients, spatial precision, depth, source alpha reference, color, four bin
tables and two metadata tensors. It saves sigma and the optional-precision flag
as scalars. Backward dispatches directly with those saved tensors and the image
cotangent. Public standalone forward and standalone direct backward APIs retain
their signatures. No native ABI, shader, binary, alpha law or threshold changes.
The two source files shrink by five lines overall.

For y_i = R(x_i; B_i, m_i), the required contribution is
J_R(x_i; B_i, m_i)^T g_i. Replacing B_i by a later render's B_j is a different
Jacobian. Saving the exact forward inputs restores this contract and removes
repeated bin packing, validation and materialization from backward. Tensor
in-place version checks remain active; this is per-graph saved state, not a
cross-update cache. Discrete support/order semantics are unchanged. Native
buffer lifetimes extend through backward and are covered by measured route
memory and the unchanged process/allocator guards.

The complete focused CPU/Metal suite passes 173 tests in 4.57 s, including both
formerly failing cases and existing geometry, alpha cutoff, interval union,
visibility, anisotropy and producer gradient coverage. Tolerances were not
relaxed. Test session 90868 returned terminal exit zero before the sweep.

## Fixed-world evidence

The sweep uses the retained 800-update, 2048-tube Coffee Martini world, cam06,
96x128 images, one-frame resident chunks, eight-frame CPU target LRU and
F={4,8,16,32} spanning the same physical interval. One warmup and three alternating
paired replay/compiled trials are retained at each F. Session 28374 returned
terminal exit zero. All four rows pass their image and world-gradient gates;
max RGB error is 8.64e-7 and max per-parameter normalized gradient error 1.65e-6.
The selected F4 slicing check passes. Retained atlas SHA and byte counts, world,
camera, target, time, active-set and tolerance contracts match the previous run.
Existing independent payload/storage/memory/timing/slicing validators pass.

E+B below is the median sum of evaluator-forward and backward from each same
trial; it excludes target loading, transfer and loss construction, but includes
Python setup, dispatch and replay projection. Full compiled cost adds compilation
and the measured I/O/loss phases. Values are seconds. Before and after are separate
processes, not an interleaved old/new binary experiment; host variation can affect
the difference, especially forward. No samples were removed.

| F | Compiled E+B before | Compiled E+B after | Improvement | Replay E+B after | Full compiled cost after |
| --- | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.24958 | 0.14850 | 1.68x | 0.03456 | 1.16003 |
| 8 | 0.57788 | 0.44284 | 1.30x | 0.07229 | 3.03137 |
| 16 | 1.70286 | 1.32884 | 1.28x | 0.25845 | 9.61079 |
| 32 | 4.05433 | 3.21791 | 1.26x | 0.53457 | 17.80326 |

F32 backward falls 1.25911 -> 0.45747 s (2.75x faster). Evaluator-forward is
2.75884 s, compilation 8.18933 s and CPU target loading 6.30534 s as separately
summarized medians. Full compiled cost falls 18.46540 -> 17.80326 s (3.59% less
time); reducing backward is now a small part of total cost. A sum of separate
medians need not equal the median of measured sums. Replay E+B remains 0.53457 s,
so the compiled evaluator plus backward is still about six times slower.

Compiled E+B grows 21.67x for 8x frames; replay grows 15.47x. This finite density
curve supports no sublinear total-cost or asymptotic-exponent claim. Removing
fixed/lower-order work improves every absolute row while increasing the growth
ratio. Local correctness remains 4/4; public counts remain 0/7 contexts and
0/21 lanes, publication_eligible is false, and BASELINES is unchanged.

## Resources, verification correction and next evidence

The sweep's sampled peak tree-plus-launcher RSS is 1,868,693,504 bytes (~1.74 GiB).
Host swap increases 1,384,120 bytes (~1.32 MiB), below the unchanged 268,435,456-byte
limit. No guard trips. The focused test peaks at 839,499,776 bytes with zero new
swap. Compiled F32 sampled MPS current/driver allocations are 3,931,392/29,343,744
bytes in the route-scoped measurement; these are sampled, not hardware peak
counters. The real host is 24 GiB, not a claimed physical 8-GiB measurement.

The copied artifact-validation application initially assumed zero new swap,
because the preceding run had zero. It failed that assertion. The failed script
and log remain with .zero_swap_assumption suffixes. The corrected application
first checks that every resource limit is identical to the preceding receipt,
then checks growth against its original 256-MiB bound. No execution guard or
scientific acceptance threshold was changed. Successful validator session 36816
is terminal exit zero; the failed session 2559 is retained. This result must not
be described as zero-swap or perfectly isolated performance evidence.

outputs/benchmarks/2026-09-13_retained_backward_inputs/ retains before/after source, baseline failure, tests, complete rows and
atlases, phase samples, verifier application, native identity and resources.
Offline W&B fr7bmn09 contains the identical report. Native SHA remains
6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.

Next use bounded Python profiling at F32 to separate remaining forward packing,
frame slicing and native execution before another optimization. Compilation
(~8.19 s) and loading (~6.31 s) must stay visible in the total-cost ledger. Do not
retry the Kineto shape profiler or infer a kernel speedup from cProfile alone.
The successful old source fits and the shared-world 20.84/15.30-dB quality run
are unaffected: this turn repairs gradient state and execution cost, not fitting
quality. Root integration owns the two regressions, this note, scoped status
updates and the STAR gitlink. Unrelated benchmark/browser/paper WIP is preserved.
