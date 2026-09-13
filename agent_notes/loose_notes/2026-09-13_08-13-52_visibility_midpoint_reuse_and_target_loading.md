# Exact visibility midpoint reuse and a measured target-loading opportunity

The preceding goal turn was progress: root f526265 / STAR 48148ac committed
exact support-bound batching and retained a matching four-row Metal sweep.
This turn refreshes the existing bounded profile, fixes its measured visibility
cost, and tests CPU target grouping. One lead, no subagents, sequential runtime
jobs, offline W&B, unchanged resource guards, no new token budget. The goal
remains active; public evidence counts stay 0/7 contexts and 0/21 lanes.

## Refreshed profile

outputs/benchmarks/2026-09-13_f32_post_bounds_profile/ retains the same F32
world/camera/target/atlas contract with synchronized Python cProfile, without
Kineto or shape/stack recording. Existing independent validators accept exact
atlas bytes, image/VJPs, source/native identity and resource receipts. Coverage
is five compiles and 160 actual slices, forwards and native backwards each.
Archived source hashes were verified before the following edit.

Accumulated diagnostic profile times: compile 18.233 s, including visibility
stratification 10.593 cumulative / 6.895 self; support rebinning 4.946 cumulative;
slicing 6.244 total, including 1,016,660 dataclass replacements at 2.651 cumulative;
forward 6.512, including 3.685 waiting for MPS and 2.115 packing; native backward
1.946, including 1.808 waiting. These instrumented sums are not ordinary benchmark
wall timings. Profile peak tree/launcher RSS is 1,906,900,992 bytes, zero new swap.
Offline W&B xbkqe2hp. The stopped shape-recording profiler was not retried.

## Exact implementation and regression

STAR f735898 extends the existing per-compilation (trace_id,start,stop) depth
cache to hold both its midpoint value and sampled min/max interval. A tile
computes float32 midpoint values only for uncached trace/span queries, then
sorts the same (depth,trace_id) pairs and emits the same depth intervals. It
does not allocate the cross-product of all traces and possible intervals.
The table is local to one compiler call, so later geometry or time changes
cannot reuse its entries. Existing active-span validation and root handling
are unchanged. There is no new shader, native ABI, retained atlas field,
precision, alpha, gradient or camera law. This is exact query reuse, not new
mathematics or an asymptotic rasterization claim.

The midpoint time remains the Python-float endpoint average and the value
remains c0 + c1*t + c2*t*t with the original float32 operation order. It must
not be substituted with the dense evaluator's c2*(t*t) expression, which can
round differently. The existing visibility test now covers partly overlapping
trace sets, distinct tile spans, two roots, and changed geometry. The focused
CPU/Metal suite remains 184 tests and passes in 4.10 s.

The CPU comparator extracts the old function from its exact retained source,
reconstructs support from retained coefficients, and compares every visibility
cell field at F4/8/16/32. Its regenerated CPU support is shared by both routes;
it is not asserted to equal the original MPS support. All cell fields match.
With one warmup and three alternating paired F32 trials, visibility falls
1.469569 -> 1.013691 s (1.450x).
Old samples: [1.440853584004799, 1.5529185000050347, 1.4695687079947675].
New samples: [1.013690708001377, 0.949773375003133, 1.0161772919964278].
Before-source SHA: 4c96371416fb05470ad706ddef1b0deb7040065c651a8030e0cf39425966d607.

## Full Metal sweep

All four local rows pass at the same 2048-tube 800-update checkpoint, cam06,
96x128, fixed physical interval, one-frame device chunks, CPU target LRU8,
one warmup and three alternating replay/compiled repeats. Atlas bytes and
world/camera/target/contract identities match exactly. Maximum RGB error is
8.64267349e-07; worst normalized per-parameter VJP error is
1.76231552e-06. Selected-time slicing passes.

Seconds below compare separate full-run processes. E+B excludes target loading,
transfer and loss but includes host setup. Full cost includes compilation and
forward/backward segments, not an optimizer step or inter-segment cleanup.
Do not causally attribute small unrelated-phase changes to midpoint reuse.

| F | Compile before | After | E+B before | After | Full before | After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.30460 | 0.29421 | 0.13488 | 0.14639 | 0.44201 | 0.44385 |
| 8 | 0.79939 | 0.72904 | 0.31393 | 0.32414 | 1.10261 | 1.06859 |
| 16 | 1.46587 | 1.24308 | 0.99948 | 0.99235 | 5.60399 | 5.37078 |
| 32 | 2.40307 | 1.84470 | 2.39479 | 2.36214 | 11.11530 | 10.55377 |

F32 compilation improves 23.24%, full cost 5.05%.
F4 full timing is slightly worse, and small-frame E+B is mixed. F32 replay
E+B remains 0.53501 s; compiled E+B grows 16.136x for 8x frames.
No sublinear or replay speed win follows. Source/shared-world quality is unchanged
because no optimizer fit ran. Local rows stay 4/4, public counts unchanged,
publication eligibility false and BASELINES unchanged.

Full-run RSS is 2032779264 bytes (1.893 GiB), new swap 0 bytes, no guard trip.
Regression and CPU visibility peaks are 910,327,808 / 575,864,832 bytes, both
zero new swap. All existing 3-GiB RSS, 2-GiB MPS, 256-MiB new swap, 600-second
and host/disk/output guards remain. Offline W&B 8roys3f5.
Native SHA stays 6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.

## Bounded target-loading control and next action

F32 target loading alone remains 6.30112 s. Code inspection locates the
cause: PaperMulticamTargetProvider groups misses within a request, but frozen
one-frame consumption makes a new VideoCapture/open/seek request on each miss.
The canonical decoder already supports grouped requests and sequential decode.
No dataset, camera, sampler, loader or benchmark code was changed for this probe.

compare_target_loading.py tests existing select_view_frames with request sizes
1 and 8 on the same ordered 32 heldout images and the same eight-frame LRU.
It hashes decoded float32 bytes incrementally, without retaining the whole clip.
All warmups/trials produce identical bytes; decoded/requested counts stay 32.
Video-open/decode calls fall 32 -> 4, and peak cached frames stay at most eight.
One warmup per size and three alternating paired trials, including byte hashing,
measure 6.217914 -> 1.118203 s (5.561x).
Raw samples: {'1': [6.186301333000301, 6.21791404200485, 6.300882833005744], '8': [1.116190791988629, 1.1182026250025956, 1.1323494579992257]}.
Target byte SHA: 533ee9f55683c6d9e99e68590030c1b20e07ea7fc877c8bc7d31063e3eac7668.
The eight-frame request increases temporary CPU batch storage; accounting is
retained rather than pretending the cache cap is a total-memory bound.
Measured CPU-run RSS is 1,001,570,304 bytes with zero new swap, under unchanged
guards. Offline W&B 3ae3knx0. The existing validator application
checks receipts, source hashes, W&B backing, exact digest equality, trial medians,
request/decode counts and both cache and decode/request batch bounds.

This is a CPU-only opportunity, not an achieved renderer speedup. Next integrate
bounded grouping/lookahead into both frozen replay and compiled consumers while
keeping one-frame Metal residency and the eight-frame CPU cache. Charge prefetch
work inside the same target_cpu_load timer, retain actual frame order and byte
identity, and include all temporary CPU buffers in accounting. Do not widen the
cache, hide loading before the timer, retain the full video, or attribute a shared
loader improvement to World Tubes' rasterizer. The grouped loader already exists;
use it rather than creating another decoder or persistent reader. Audit current
provider tests under test_powerfoam_target_provider.py and the frozen runner.

All artifacts are in outputs/benchmarks/2026-09-13_visibility_midpoint_reuse/. Source and native hashes
were verified and archived. Integration owns only the source leaf, strengthened
visibility test, this note, scoped maps/learning edits and STAR gitlink. Unrelated
root and benchmark WIP stays unstaged. This turn's demonstrated improvements and
new measured loader result change the next action; the broad goal remains active.
