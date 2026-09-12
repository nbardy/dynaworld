# Batch support bounds without changing float32 tile decisions

The preceding committed optimization was exact one-frame interval packing
(root 0bcd31c / STAR 44c8265). Its bounded F32 Python profile retained 158,370
scalar quadratic-bound calls, 2.406 s cumulative over five compilations. This
turn completes the pending batching change against the same saved learned world.
One lead, no subagents, sequential accelerator jobs, unchanged local resource
guards, offline W&B, and no new token budget. The overnight goal remains active.

## Convergence clarification

The user asked whether successful source overfits had been forgotten. Re-reading
the raw May JSONs confirms 29.82337 dB / 0.85720 SSIM at 256px and 29.13822 /
0.86059 at 512px. The July source recipe was already reproduced September 12:
21.768529 -> 21.768527 dB, with scientific config sections identical. The saved
July media comparison is STAR 21.807, dynamic 3DGS 18.643, WorldFoam 17.777 dB,
but has different primitive counts and updates. Existing notes
2026-09-12_15-08-26_source_overfit_vs_small_multicam_correction.md and
2026-09-12_15-32-00_world_tube_overfit_controls_and_initialization_fix.md preserve
that evidence and its limits. The inspected May contact sheet remains a good
source fit with softened detail.

The old fitter optimizes ScreenTimeTubeModel with full-clip MSE. The new
WorldTubeModel shares 3D parameters across two calibrated cameras and samples
two target images/update with robust L1. The initial 256-tube world run failed
on train too (6.49/6.50 dB), while the corrected 2048-tube 800-update world
reaches 20.84/15.30 dB. Both current previews remain blurry; the heldout camera
is markedly worse. No new optimizer fit ran this turn. Poor shared-world results
do not revoke the reproduced source-fit result, and none proves exact pixel
convergence. Reconstruction and compiler-performance claims stay separate.

## Implementation and numerical contract

STAR 48148ac adds batched endpoint/vertex bounds and uses them in support rebinning.
The original trace/root loops collect (trace,start,stop) spans in their existing
order. CPU tensor gathers evaluate all U/V/depth quadratics together, then the
unchanged tile-box and record assembly logic consumes each result in order.
These transient tensors are proportional to the number of spans; no atlas
storage, native ABI, shader, camera, opacity or gradient law changes.

Each polynomial remains c0 + c1*t + c2*t*t in the original float32 operation
order. Vertex coordinates use float64, matching the old Python-float division;
interval inclusion happens before casting the vertex to float32 for evaluation.
Strict comparison/select retains the first endpoint on ties or unordered
comparisons, including signed-zero behavior. Zero quadratic terms exclude
computed inf/NaN vertices. The old scalar helper remains as an exact reference.
This is vectorized evaluation of ordinary quadratic extrema, not new gauge math
or an asymptotic complexity change.

The focused CPU/Metal suite passes 184 tests in 4.22 s. The added test checks
integer-bit equality against the scalar reference at endpoints, interior
vertices, zero/subnormal coefficients and signed zero, plus empty input.
The CPU comparator loads independently verified F4/8/16/32 atlases and checks
all fields after old/new rebinning. Per-trace alpha radii are reconstructed on
CPU once and shared between routes; this is not a claim that reconstructed
radii equal the original MPS calculation. The full Metal sweep below separately
checks actual production atlas bytes. All four CPU atlas comparisons pass.

Paired F32 CPU rebinning, including span tensor setup, with one warmup per route
and three alternating repetitions: 1.057954 ->
0.705755 s (1.499x).
Old samples: [1.057953583993367, 1.1530405840021558, 1.041310832995805].
New samples: [0.7057553339982405, 0.6552362499933224, 0.7122242920013377].
Retained before-source SHA: 7effb2af57fdfe39ab6cb86c70f51491e977db7eb7083d9d798d7585b5a89719.

## Full Metal measurement and independent verification

All four local rows pass using the same 2048-tube 800-update checkpoint,
cam06, 96x128 targets, fixed physical interval, one-frame chunks, target LRU8,
one warmup and three alternating replay/compiled repetitions. Every retained
atlas byte and the bound world/camera/target/contract fields match the preceding
run. Maximum RGB error 8.64267349e-07; maximum normalized
per-parameter VJP error 1.81920963e-06. Selected-time slicing passes.
The existing independent contract validators accept the reports, source/native
hashes, resource receipts and offline W&B backing. No tolerance was changed.

Seconds below compare separate full-run processes. Evaluator+backward excludes
target I/O/transfer/loss but includes host setup and replay projection. Full cost
includes compilation and forward/backward segments, not an optimizer step or
inter-segment cleanup. Do not causally attribute small unrelated-phase timing
differences to this edit. The CPU comparison above is paired within one process.

| F | Compile before | After | E+B before | After | Full before | After |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4 | 0.42869 | 0.30460 | 0.14254 | 0.13488 | 0.57204 | 0.44201 |
| 8 | 1.05698 | 0.79939 | 0.30461 | 0.31393 | 1.36543 | 1.10261 |
| 16 | 1.77933 | 1.46587 | 0.94451 | 0.99948 | 5.84786 | 5.60399 |
| 32 | 2.88718 | 2.40307 | 2.33777 | 2.39479 | 11.52976 | 11.11530 |

F32 compile change is 16.77%; full measured cost change
3.59%. Current target loading is 6.28949 s, and replay
E+B is 0.52180 s. Compiled E+B grows
17.755x for 8x frames. Separate phase medians need not sum to the median of
same-trial totals. This does not establish sublinear rasterization or a replay
speed win. Local numerical acceptance stays 4/4; public counts stay 0/7 contexts
and 0/21 lanes. Publication eligibility remains false; BASELINES is unchanged.

## Resources, retained artifacts and next work

Full-run peak tree plus launcher RSS is 2051358720 bytes (1.910 GiB), new
swap 0 bytes, no guard trip. Regression and CPU-comparison peaks are
907,329,536 and 644,087,808 bytes, both zero new swap. All existing 3-GiB RSS,
2-GiB MPS allocator, 256-MiB new swap, 600-second and host/disk guards remain.
No app kills, online uploads, remote jobs or stopped shape-recording profiles.

Offline W&B nrhn3nwr. Native SHA 6995d093694fda0e54c85926aa94bd3f3aeab58019c4121427b54dd2b8559cf5.
outputs/benchmarks/2026-09-13_batched_quadratic_bounds/ retains before/after sources, comparison
script/results, test log, all raw timing samples, exact atlas artifacts, resource
receipts and the existing independent validator application. Source and native
hashes are checked before integration. Only the owned source leaf, test, note,
scoped shared status/learning edits and gitlink are committed; unrelated WIP stays.

The preceding bounded profile still locates substantial visibility-event and
slice construction costs. Refresh that existing bounded profile after these
compile/packing changes before selecting another implementation change; do not
retry Kineto/record_shapes or infer GPU saturation from total forward time.
Keep the successful source-fit controls and the remaining shared-world quality
gap explicit while measuring renderer progress.
