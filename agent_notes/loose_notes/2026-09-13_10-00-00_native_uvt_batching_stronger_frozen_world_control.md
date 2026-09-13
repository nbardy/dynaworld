# Ordinary native UVT batching is the stronger static-world control

The preceding turn rechecked and explained historical successful source fits;
it repeated existing evidence and added no runtime result. This continuation
resumed the unfinished ordinary-native batching diagnostic. Three execution
layouts now pass independent verification. The result changes the next action:
use native batching as a stronger static affine control before spending more
time on interval-compiler micro-optimizations. One lead; no subagents, no
concurrent accelerator jobs, native rebuilds, app termination or online upload.

## Frozen experiment

Same Coffee Martini cam06, 32 contiguous selected dataset times, 96x128,
2048 legacy world tubes from the completed 800-update run. The checkpoint SHA
is da6eac63e2497a47fe7bcfa5be645b7b665199c318e17384b31cbf493f4e0c77;
logical world SHA is
2ac9be4f45e1304d9ed2e8208303d3ed2ccf3a1713d3cd1ab89be6b46233434e.
Each route uses ordinary project_world_tube_sequence and native
render_projected_sequence, peak_splat, direct_atomic/index_add, tile_t=1,
capacity 256. Native frame batch sizes are 1, 4 and 32. Full-clip time is
preserved by the existing ma.z recentering. This is a static camera control
on the existing affine UVT model, not exact perspective motion or an orbit.

The CPU target LRU and each decode/request stay at eight frames. All layouts
consume identical ordered target bytes and the actual robust-L1 objective,
sqrt(error^2+1e-6), normalized by the same global element count. Larger batches
intentionally retain more rendered output and loss residuals on Metal; those
allocations are included. No full video target tensor is constructed.

One correctness pass per layout precedes timing. One warmup per layout is
followed by three trials, rotating order [1,4,32], [4,32,1], [32,1,4]. These
are synchronized host-plus-device phase measurements, not GPU kernel times.
Projection, binning, rendering and backward are charged. Full measured cost
also includes target decode, transfer and loss; it excludes startup, checking,
hashing, logging, cleanup between segments and any optimizer step. The process
receipt includes the entire launched process tree and offline W&B helper.

## Results

| Native frame batch | Projection/bin/render + backward | Full measured phases |
| ---: | ---: | ---: |
| 1 | 0.33040 s | 1.46145 s |
| 4 | 0.16795 s | 1.28540 s |
| 32 | 0.05955 s | 1.18075 s |

Batch 32 is 5.548x faster for evaluator/backward and 1.238x for all measured
phases than batch 1 in this same-process rotating comparison (81.98% and
19.21% less time respectively). Target decode stays about 1.11 s and dominates
the full result. This does not make a claim about optimizer throughput.

For context, the prior interval-atlas F32/chunk4 result measured 1.18709 s
evaluator/backward and 4.18126 s including compilation and other phases.
Those are 19.93x and 3.54x the new native-batch 32 medians, respectively. This
is a separate-run comparison, not a randomized direct timing pair. Target and
camera hashes, checkpoint, world state, loss and numerical tolerances match;
the interval topology and batching algorithm intentionally differ. Ordinary
native UVT preserves its temporal tiling, so the previous interval-packer's
whole-chunk candidate-union capacity failure does not apply automatically.
The ordinary native forward rejects any tile overflow before returning RGB.

## Verification and failures retained

The first attempt stopped before rendering because the new runner indexed
absent lens metadata. The fix uses the existing camera/lens selectors, also
selecting the first camera transform correctly from [view,time,4,4]. Attempt 02
stopped on the native import-time tile_t/config mismatch. The runner now sets
the wrapper's existing specialization environment from retained training
settings before importing it. Neither error was a renderer-quality negative.
Their logs, failed reports, W&B identities, receipts and exact bound sources
remain in the parent output folder and attempt02/failed_sources.

Attempt 03 completed, offline W&B 3kk2mb2v. Independent validate_results.py
reloads seven-frame CPU target chunks, recomputes loss and all seven parameter
gradient comparisons with NumPy float64, checks finite values/shapes/coverage,
recomputes raw timing summaries and verifies source/native/checkpoint hashes,
world nonmutation, memory receipts and offline artifact backing. It uses the
existing frozen-world 1e-5 image/loss/gradient thresholds without relaxation.

Maximum RGB difference is 2.98023224e-7 for batch 4; batch 32 is exactly equal
to batch 1. Worst global normalized gradient error is 7.28605846e-6 and worst
per-parameter normalized error is 5.55291233e-6. The common recomputed loss
is 0.12422011682, matching the retained previous evaluator within tolerance.
Native batch 32's correctness result is one atomic-backward sample; this does
not establish a bound on repeated nondeterministic reduction error.

The first independent checker tried CPU camera construction and failed exact
camera identity. Repeating the producer's Metal calibration arithmetic makes
the hash exactly b7356f010e88ecacc2bae87e3dc6f9275cee6b63e1fc59d6c5e1dcafba8df796.
The independently reconstructed CPU w2c differs from Metal by at most
2.384185791015625e-7; K is exact. The failed checker/log/source set is retained
as cpu_camera_failure and cpu_verifier_failure_sources. This is an arithmetic
backend difference, not permission to weaken the camera hash. Target identity
matches the old NHWC contract exactly after accounting for its dtype/shape
header; the producer's new raw NCHW digest is checked separately.

Peak tree-plus-launcher RSS is 1,109,803,008 bytes (1.034 GiB); verification
peaks at 804,306,944 bytes. Both have zero new swap and no guard trip. Maximum
sampled MPS current allocation during correctness is 22,901,760 bytes; timing
driver allocation peaks at 97,452,032 bytes. These are sampled observations
on the 24-GiB physical Mac, not an 8-GiB-hardware result. All existing 3-GiB
RSS, 2-GiB allocator, host/disk/swap and 600-second guards remain unchanged.

No existing renderer changed, so no native rebuild or broad regression rerun
was necessary. The actual three-layout Metal run exercises the new runner's
projection, rendering, target loading, backward, logging and saving. Successful
run-bound source sets are archived under attempt03/after and hash checked.
The broader worktree remains dirty; this is not a clean-checkout reproduction.

## Reproduce and next scientific decision

Tracked runner: research_experiments/paper_runner_suite/compare_native_uvt_batches.py.
Config: src/train_configs/frozen_world_native_uvt_control_20260913.jsonc.
The successful launch used attempt03/resolved_runtime_config.json, changing
only output_dir from the checked-in config. Preserve existing artifacts: for
a fresh rerun, choose a new output directory and retain the equivalent launcher,
resolved config, native identity and source bindings. The original command was:

    PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python outputs/benchmarks/2026-09-13_native_uvt_batch_control/attempt03/launch.py evaluate_native_batches.py

The independent check uses the same launcher with validate_results.py. Neither
command should be rerun over retained evidence without fresh output paths.

For fixed F and batch B, the number of world projections changes from F to
ceil(F/B). B=F projects once while native bins, compositing, image writes and
the image-loss adjoint still perform frame-dependent work. This is ordinary
batching of an already time-parametric representation, not new gauge math.
Holding F=32 fixed while changing B estimates no exponent in F. A complete
image sequence still has F*H*W outputs. Native batching must be a comparator
where its time/camera assumptions hold; per-frame replay alone is too weak to
attribute an advantage specifically to an event atlas.

Next extend this stronger control to the fixed-world selected-time sweep.
One exact candidate is to render the full native 32-frame lattice, gather only
the requested times for the same loss, and charge every extra rendered frame,
output byte and backward operation. This handles the existing nonuniform
integer selections without silently changing targets to a convenient uniform
grid. First check its selected-loss world VJP against replay; then compare
runtime with the interval atlas. It is a local 32-frame control, not a general
long-video or moving-camera solution. General camera/event benefits still need
their own measured evidence. Do not infer a sublinear title from this result.

Local native layouts advance 0/3 to 3/3 verified. Earlier interval layouts stay
5 passing and 3 capacity-negative, fixed-density checks 4/4, public acceptance
0/7 contexts and 0/21 lanes. Source-fit quality is unchanged. BASELINES gains
explicit local execution-control rows; no publication standings are promoted.
The broad overnight goal remains active.
