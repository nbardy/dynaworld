# Browser WebGPU Kernel Forks: Scientist Reflection

Date: 2026-07-31 KST

## 1. Context

This note records the July 30-31 browser-kernel fork, including the model that
motivated it, the implementations that survived measurement, the ones that did
not, and the remaining falsification work.

The trigger was a real discrepancy:

- the browser could train a calibrated full image, but 20K-30K splats were much
  slower than remembered Metal microbenchmarks;
- the old browser backward repeated expensive 3D projection/covariance VJP
  work inside the tile/splat loop;
- the quality plateau and the speed problem were being discussed together even
  though they are different hypotheses;
- manual browser benchmarking made repeated forks awkward and encouraged
  one-shot timing claims.

Repository boundary:

- `web/dynaworld_browser_trainer/` remains a demo and systems prototype;
- it consumes the canonical exported multicamera/calibration/split contract;
- it does not create a WebGPU SfM stack;
- it does not merge WebGPU abstractions into the Python trainer hierarchy;
- its current model is trajectory-gated dynamic 3DGS, not native 4DGS and not
  World Tubes.

Relevant files:

- `web/dynaworld_browser_trainer/trainerWebGpu3dTiled.js`
- `web/dynaworld_browser_trainer/trainerWebGpu3dTiledFast.js`
- `web/dynaworld_browser_trainer/benchmarkTiledKernels.js`
- `web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js`
- `web/dynaworld_browser_trainer/tiledParityHarness.js`
- `web/dynaworld_browser_trainer/benchmark_results/2026-07-31_interleaved/`
- `web/dynaworld_browser_trainer/benchmark_results/2026-07-31_wgpu_kernel_forks_apple_m4.json`

## 2. Evidence Labels

This note uses these labels deliberately:

- **Observed:** directly measured in code, a test, a GPU parity run, or a saved
  artifact.
- **Inferred:** the simplest current explanation of observations.
- **Speculative:** plausible but not yet measured.
- **Proposed:** a concrete next experiment or implementation.

This matters because the browser quality plateau, the Metal recollection, and
the kernel timings do not all have the same evidence strength.

## 3. Current Working Model

### 3.1 Representation

**Observed:** each primitive has a base 3D center, velocity, optional harmonic
offset, temporal gate/static mix, anisotropic 3D covariance, quaternion,
constant RGB, and opacity. It projects a 3D covariance through a calibrated
pinhole camera at the selected time.

**Inference:** the precise current representation name is
`trajectory-gated dynamic 3DGS`.

**Not supported:** calling the current path native spacetime Gaussian splats,
native 4DGS, or World Tubes. Those require different representation and
backward contracts, not just a UI label.

### 3.2 Fast step dataflow

The promoted `tiled3d-fast` step is:

```text
project all active splats
    -> 32 B/splat hot raster packet
    -> 80 B/splat cold VJP packet
bin exact ellipse support into 8x8 tiles
depth sort each tile
front-to-back source-over raster with packed transmittance checkpoints
exact separable 11x11 Gaussian SSIM forward
exact separable SSIM transpose gradient
checkpoint-block raster backward
    -> 12-float projected gradient/splat
one 3D projection/covariance/trajectory VJP per splat
Adam update
optional bounded topology event
```

The direct `tiled3d` reference retains the older monolithic projection packet
and repeated 3D VJP. Both remain runnable so "fast" does not become an
unfalsifiable rewrite.

### 3.3 Headless benchmark architecture

The benchmark is a Bun command, not a human-driven webpage:

```text
Bun
  -> starts private no-store HTTP server
  -> launches private headless Chrome
  -> Chrome/Dawn exposes navigator.gpu
  -> benchmark module runs and returns structured JSON
  -> Bun writes artifact
  -> Bun closes Chrome and server
```

**Observed:** Bun does not expose `navigator.gpu` or the browser WebGPU object
model. A WebGPU runtime is therefore still needed.

**Decision:** Chrome is a hidden device runtime dependency. The HTML page is
an implementation module and optional visual debugger, not the benchmark user
interface. This answers the objection that kernel testing should not require
opening and operating a webpage.

## 4. Assumptions And Ranges

The measured claims are bounded by:

| Variable | Measured range |
| --- | --- |
| GPU | Apple M4 through WebGPU |
| Raster | 96x72 and 192x144 |
| Active splats | 8,192 and 30,000 headline rows |
| Cameras/times | Coffee Martini train17, holdout1, 16 times |
| Tile edge | 8 promoted, 16 retained |
| Tile capacity | up to 4,096 contributors |
| Checkpoint stride | 16 promoted |
| Checkpoint storage | packed FP16 promoted |
| Arithmetic | FP32 |
| Projection storage | FP32 |
| Gradients/moments | FP32 |
| Objective | 0.8 L1 + 0.2 DSSIM |

Important limits:

1. Packed FP16 applies to transmittance checkpoint storage, not native FP16
   shader arithmetic.
2. The 30K row is near a scene-specific tile-capacity boundary.
3. The 32K full camera/time cycle is invalid because cumulative tile overflow
   is nonzero.
4. A speedup at fixed steps does not prove a better reconstruction at fixed
   wall time.
5. No saved Metal artifact currently matches the complete browser step,
   objective, raster, splat count, and synchronization protocol.

## 5. Derivation A: Why Stage The 3D VJP?

Definitions:

- `N`: active scene splats.
- `P`: active tile/splat pairs after binning.
- `d_r`: cost of raster-space derivative work.
- `d_3`: cost of projection, covariance, quaternion, scale, and trajectory VJP.
- `a`: cost of atomic accumulation into a projected gradient.

The old direct structure approximately pays:

```text
C_direct ~= P * (d_r + d_3)
```

The staged structure approximately pays:

```text
C_staged ~= P * (d_r + a) + N * d_3
```

The avoided work is:

```text
Delta ~= (P - N) * d_3 - P * a
```

Staging should win when:

```text
(P/N - 1) * d_3 > (P/N) * a
```

This is especially plausible when a splat touches multiple tiles, because
`P/N > 1`. The gain need not grow without bound: at high splat count, projection,
sorting, clearing, and update become a larger fraction of the complete step.

**Observed:** staged backward improved end-to-end throughput by 1.84x at 8K and
1.70x at 30K. The reported backward phase improved by 2.52x and 2.79x.

Profiler caveat:

- direct `backward` includes raster derivatives and repeated 3D VJP;
- staged `backward` includes raster derivatives and projected accumulation;
- staged `update` includes one 3D VJP per splat plus Adam;
- direct `update` includes Adam only.

The backward ratio localizes the removed work but is not a comparison of
identical kernel boundaries. End-to-end throughput and complete GPU span are
the primary evidence. The profiler now emits both summed pass time and
first-begin to last-end `gpuSpanMs`, along with this phase contract.

**Inference:** the structural model is supported. The smaller end-to-end gain
at 30K means non-backward phases now matter more; it does not refute the
backward optimization.

## 6. Derivation B: Why Checkpoint-Block Replay?

Let `B` be the number of sorted contributors between two transmittance
checkpoints for one pixel.

In a pair-owned reconstruction, contributor `i` may replay the earlier or later
part of the interval to recover its source-over state. Summed across the block,
the traversal count can approach:

```text
1 + 2 + ... + B = B(B + 1) / 2 = O(B^2)
```

A checkpoint-block workgroup instead walks the interval once per pixel lane,
updates the running prefix/suffix state, and emits all `B` projected
contributor gradients:

```text
B contributor visits + B accumulation writes = O(B)
```

The exact constant depends on visibility termination, lane occupancy, and
atomics. The asymptotic model does not promise a win for tiny `B`.

**Observed:** checkpoint-block plus staged VJP survived the measured fork.

**Backtrack:** a shared pair packet intended to reduce global traffic was
slower. Workgroup sharing is not automatically beneficial; synchronization,
occupancy, and extra copies can dominate.

## 7. Derivation C: Hot/Cold Projection Packets

The old monolithic projection record is 192 bytes per splat.

The split layout is:

```text
hot raster packet: 32 bytes
cold VJP packet:   80 bytes
total:            112 bytes/splat
```

The cold packet contains:

```text
camera point + valid marker        16 B
sparse pinhole Jacobian terms      16 B
three basis-vector/variance lanes  48 B
                                  ----
                                   80 B
```

Camera rotation is shared camera state and is reconstructed in the update
kernel rather than copied into every splat record.

Storage saving:

```text
Delta_bytes(N) = (192 - 112) * N = 80N
```

At 30,000 splats:

```text
80 * 30,000 = 2,400,000 bytes = 2.29 MiB
```

The hot stages now fetch only 32 bytes rather than 192:

```text
hot-read reduction = 1 - 32/192 = 83.3%
```

This does not mean an 83% complete-step speedup because sort/raster also read
other buffers and because the cold packet is still consumed once per update.

**Observed:** two reversed-start 30K runs improved throughput by 1.033x and
1.062x; mean GPU-step speed improved 1.072x.

**Inference:** the layout is worthwhile primarily as a bandwidth and memory
headroom improvement. Its speed benefit is real but modest.

## 8. Derivation D: Exact Separable SSIM

Let `h` be the normalized 11-tap Gaussian with sigma 1.5. Reflection at the
image boundaries is part of the operator. Define:

```text
H_x: horizontal reflected convolution by h
H_y: vertical reflected convolution by h
H = H_y H_x
```

For prediction `x` and target `y`, local moments are:

```text
mu_x = H(x)
mu_y = H(y)
E_x2 = H(x^2)
E_y2 = H(y^2)
E_xy = H(xy)
sigma_x2 = E_x2 - mu_x^2
sigma_y2 = E_y2 - mu_y^2
sigma_xy = E_xy - mu_x mu_y
```

Per-channel SSIM is:

```text
A = 2 mu_x mu_y + C1
B = 2 sigma_xy + C2
C = mu_x^2 + mu_y^2 + C1
D = sigma_x2 + sigma_y2 + C2
SSIM = (A B) / (C D)
```

with `C1 = 0.01^2` and `C2 = 0.03^2`.

For a loss derivative with respect to the moment images, the convolution
adjoint must be applied in reverse order:

```text
H^T = (H_y H_x)^T = H_x^T H_y^T
```

Reflection makes boundary indexing easy to get wrong. The forward and adjoint
passes therefore remain covered by CPU value and finite-difference checks plus
the live GPU parity harness.

Tap-work model per scalar field:

```text
direct 2D forward + adjoint: 121 + 121 = 242 taps
separable forward + adjoint: 11 + 11 + 11 + 11 = 44 taps
```

This is a 5.5x tap-count reduction before intermediate-memory traffic and pass
launch overhead. The implementation allocates 80 scratch bytes per pixel:

```text
96 * 72 * 80 =   552,960 bytes
192 * 144 * 80 = 2,211,840 bytes
```

**Observed:** SSIM stats improved 2.50x-2.65x; the adjoint improved
3.66x-5.70x; end-to-end throughput improved 1.05x-1.28x across measured
workloads.

**Inference:** the larger full-step gain at 192x144 is consistent with SSIM
becoming a larger fraction of the raster-size-dependent work.

## 9. Why 8x8 Tiles Won Here

Changing 16x16 to 8x8 does several things at once:

- reduces pixel lanes per workgroup from 256 to 64;
- reduces wasted pixel work near a splat's support boundary;
- increases the number of tiles and duplicated splat/tile references;
- changes workgroup occupancy and scratch pressure;
- changes the distribution of contributors per tile.

There is no universal theorem that 8x8 is faster. The promoted value is a
measured workload choice, not an API invariant.

**Proposed falsification:** rerun 8x8 versus 16x16 on at least one second scene,
192x144, and both sparse and dense occupancy. If the winner changes, choose by
device/workload class rather than fixing one global value.

## 10. Measurement Protocol Backtrack

### Prior model

Run a control trainer, destroy it, run a candidate trainer, and compare the two
wall-clock intervals.

### Status

**Invalidated for headline claims.**

### Evidence

Reversing process/variant order materially changed apparent margins. Shader
compilation, device warm state, browser scheduling, and GPU frequency state can
all contaminate a one-shot comparison.

### Replacement protocol

1. Construct both variants and keep them alive.
2. Warm both.
3. Divide measured steps into four equal chunks.
4. Alternate execution order each chunk.
5. Alternate timestamp-profile order.
6. Drain the queue only at measured boundaries.
7. Record each round, not only the aggregate.
8. Reverse which variant starts in a second run for small expected gains.

Historical one-shot rows remain in the artifact for chronology but are not
headline evidence.

### Process-cleanup backtrack

We initially suspected several system headless-Chrome processes were leaked by
the Bun runner. PID ancestry showed they belonged to unrelated persistent
Playwright CLI daemons. The benchmark's own Chrome child was already gone.

One Bun process did remain alive after writing its result because an event-loop
handle stayed open. The runner now performs bounded cleanup of only its own
browser process and exits explicitly. We did not kill unrelated processes.

## 11. Accepted, Rejected, And Parked Lanes

### Accepted

| Lane | Evidence | Decision |
| --- | --- | --- |
| Staged projected gradient | 1.84x/1.70x throughput, parity pass | default |
| One 3D VJP per splat | 2.52x/2.79x reported backward phase, boundary caveat | default |
| Checkpoint-block backward | wins within fast fork, parity pass | default |
| 8x8 tile edge | measured winner on current workload | default, keep control |
| 32 B hot + 80 B cold projection | 1.05x at 30K, -2.4 MB | default |
| Pixel-major checkpoints | beat block-major | default |
| Checkpoint stride 16 | beat 8 and 32 | default |
| Exact separable SSIM | 1.05x-1.28x complete step | default |
| Bun headless command | repeatable JSON, no manual UI | default workflow |

### Rejected on current evidence

| Lane | Observation | Interpretation |
| --- | --- | --- |
| Shared pair packet | slower | sync/copy/occupancy cost exceeded savings |
| Checkpoint stride 8 | slower | more storage/writes exceeded replay saving |
| Checkpoint stride 32 | slower | replay cost exceeded storage saving |
| Block-major checkpoint order | slower | access pattern did not help Apple path |
| 32K/4,096 tile capacity | cumulative overflow | invalid result, not "slightly approximate" |
| One-shot variant timing | order sensitive | insufficient benchmark design |

### Parked, not disproved

- subgroup reductions and scans;
- adaptive tile size;
- parallel/radix tile sort;
- storage-only FP16 projection packets;
- active-splat compaction;
- source-over prefix/tape reuse;
- residual/depth-guided topology;
- view-dependent appearance;
- native 4DGS;
- World Tubes.

These lanes require separate tests. "Not implemented in this fork" is not
evidence against them.

## 12. Speed And Convergence Must Stay Separate

**Observed speed result:** the accepted fast kernel performs the same checked
one-step objective and update substantially faster.

**Observed quality state:** the browser run can still plateau at limited
high-frequency detail and heldout quality.

**Inference:** the kernel fork removes a systems bottleneck but does not by
itself fix representation, initialization, appearance, topology, or
optimization limitations.

Potential convergence bottlenecks include:

1. sparse or provenance-uncertain initialization;
2. fixed capacity after the initial fill;
3. no residual/depth-guided split and prune schedule;
4. constant RGB instead of view-dependent appearance;
5. simple linear/harmonic center trajectories rather than a native spacetime
   covariance or deformation field;
6. one camera/time image per step and camera/time loss variance;
7. learning-rate family balance and late schedule decay;
8. aspect-ratio trust region;
9. visibility/occlusion mistakes that a photometric objective cannot easily
   escape;
10. insufficient heldout support rather than insufficient raw splat count.

A faster bad objective is still bad. Conversely, a visually noisy per-step
loss can be camera/time sampling variance even when the cycle mean improves.

## 13. Metal Comparison: What We Can And Cannot Say

**Observed:** the new browser structure imports the useful idea from the Metal
lineage: share raster work, accumulate a compact projected adjoint, and run
expensive model-space VJP once per primitive.

**Not observed:** a complete Metal optimizer step matched on:

- 8K and 30K trainable splats;
- 96x72 and 192x144 raster;
- 0.8 L1 + 0.2 exact Gaussian DSSIM;
- same camera/time selection;
- same projection and source-over semantics;
- same topology setting;
- same synchronization boundary.

Therefore this work cannot support "WebGPU is only 20-30% slower than Metal."
That claim needs a matched harness in both runtimes. Historical million-scale
Metal counts may refer to projected instances, segments, raster-only passes, or
different backward cuts rather than one million trainable scene splats through
this complete step.

**Proposed:** define a shared binary fixture and phase schema, run the same
forward/objective/backward/update boundary in Metal and WebGPU, and compare:

```text
project_ms
bin_ms
sort_ms
raster_ms
ssim_stats_ms
ssim_grad_ms
backward_ms
update_ms
bytes_allocated
tile_pairs
visible_splats
overflow
```

Only then should a percentage claim enter a project standing.

## 14. Ranked Next Lanes

### P0: Matched wall-time quality gate

Hypothesis:
    Fast and direct tiled paths are numerically equivalent enough that faster
    steps yield equal or better quality at fixed wall time.

Cheap test:
    Same seed, same active cameras/times, fixed topology, 10K steps and fixed
    wall time; compare train/heldout L1, PSNR, SSIM, parameter deltas, and
    rendered-image max error.

Failure:
    A small per-step atomic-order difference compounds into materially worse
    quality.

Decision:
    Keep fast as default only if fixed-step quality is statistically matched;
    report fixed-wall-time separately.

### P0: Multi-scene parity and overflow

Hypothesis:
    The fast packet and replay math generalize beyond Coffee Martini occupancy.

Cheap test:
    One static-heavy and one motion-heavy calibrated scene at 8K and 30K.

Failure:
    Gradient parity or tile overflow depends on one scene's depth/support
    distribution.

Decision:
    If failures cluster by occupancy, add an occupancy-qualified config or
    adaptive tile capacity rather than hiding overflow.

### P1: Active-splat compaction

Hypothesis:
    Large reserved capacities pay avoidable clear/update/preview costs for
    dormant slots.

Cheap test:
    Profile 30K capacity with 8K active versus 30K active and report phase
    deltas. Add a compact active-ID list only if dormant cost is substantial.

Decision:
    Compact project/update/preview dispatches; preserve stable global IDs for
    optimizer and topology state.

### P1: Tile sort specialization

Hypothesis:
    After the backward fix, bin/sort becomes a dominant dense-scene phase.

Cheap test:
    Record occupancy histogram and sort time by tile. Compare current bounded
    sort with small-count insertion/bitonic specialization and a parallel
    radix/counting design.

Failure:
    Raster or update dominates; sort work is not worth architectural risk.

### P1: Residual/depth-guided topology

Hypothesis:
    The quality plateau reflects where splats are allocated, not just how many
    exist.

Cheap test:
    Freeze all optimizer settings and compare fixed topology, current fill-only
    split, and residual/depth-guided split/prune at 8K.

Required diagnostics:
    residual by tile, max projected radius, opacity mass, visibility count,
    gradient norm, split parent/child lineage, prune reason, and heldout delta.

Decision:
    Promote only if heldout quality improves, not merely train loss.

### P1: Source-over tape/prefix reuse

Hypothesis:
    The checkpoint-block path still reconstructs enough compositing state that
    a more compact prefix/tape can reduce replay without exploding memory.

Cheap test:
    Instrument contributor visits per pixel and bytes per visible pair.

Failure:
    The current packed checkpoints are already below the bandwidth/compute
    crossover.

### P2: Storage-only FP16 projection VJP

Hypothesis:
    Camera point, sparse Jacobian, and basis/variance cold fields tolerate FP16
    storage while gradients and moments stay FP32.

Cheap test:
    Pack one field family at a time. Run adversarial depth, tiny/large scale,
    high anisotropy, and near-frustum fixtures plus one-step parity.

Failure thresholds:
    any invalid projection, overflow change, gradient-family parity failure, or
    fixed-step quality regression.

Decision:
    Never pack Adam moments, projected-gradient atomics, or loss reductions
    merely to advertise FP16.

### P2: Host target memory

Hypothesis:
    At 384x288 and above, host-side decoded Float32 targets and worker copies
    become a larger memory problem than the paged GPU target.

Cheap test:
    Measure retained host bytes across main/training/validation workers.

Decision:
    Keep canonical RGBA8/packed targets and decode/page one camera/time frame
    near use; use shared immutable storage where browser support permits.

### P2: Preview and validation isolation

Hypothesis:
    The live SPA still shows burst/freeze behavior from preview sorting,
    readback, or validation even though the kernel lab excludes them.

Cheap test:
    Timestamp training command submission independently, disable each consumer,
    and compare p50/p95/p99 inter-step latency.

Decision:
    Keep training nonblocking; let UI and validation consume stale snapshots
    rather than synchronizing the train queue.

### P3: Native representation ablations

Hypothesis:
    Trajectory-gated 3D covariance is the quality limit after topology and
    appearance are controlled.

Required branches:

- calibrated dynamic 3DGS baseline;
- native 4D spacetime Gaussian covariance;
- World Tubes shared-backward representation.

Decision:
    Implement as explicit, faithful backends only after the current baseline
    has matched quality and benchmark contracts. Do not expose partial probe
    shaders as options.

## 15. Falsification Matrix

| Claim | Supporting result | Result that weakens it | Next action |
| --- | --- | --- | --- |
| Staged VJP is faster | >1.2x alternating throughput | <=1.05x on second scene | choose by occupancy/device |
| Staged VJP is correct | parity all gradient families | any systematic family drift | fix before quality runs |
| Compact packet helps | reversed-order runs both >1 | winner flips by order | demote to memory-only option |
| Separable SSIM is exact | value/gradient parity | edge/corner mismatch | audit reflection adjoint |
| 8x8 is the right default | multi-scene median win | 16x16 wins dense/high-res | adaptive selection |
| More splats fix quality | heldout improves at matched time | only train loss improves | topology/appearance focus |
| FP16 cold storage is safe | adversarial parity passes | depth/scale drift | keep FP32 |
| UI is nonblocking | stable p99 train cadence | periodic queue gaps | remove readback dependency |
| WebGPU is near Metal | matched full-step result | phase mismatch/large gap | optimize dominant matched phase |

## 16. Immediate Work Remaining

Code/system gates:

1. Keep the unit suite green.
2. Keep the active Apple parity harness green.
3. Smoke the default SPA worker path, not only the isolated kernel trainer.
4. Preserve raw JSON and exact command lines for every promoted benchmark.
5. Add a second-scene alternating benchmark before generalizing Apple M4
   margins.

Quality gates:

1. Matched direct versus fast convergence at fixed steps and wall time.
2. Verified 768-point train-only initialization plus growth versus legacy 4,096
   unverified seeds.
3. Fixed topology versus residual/depth-guided topology.
4. Multi-scene, multi-seed train and heldout PSNR/SSIM/L1/LPIPS.
5. Only then evaluate native 4DGS or World Tubes browser representation lanes.

Memory gates:

1. Quantify host target duplication.
2. Quantify dormant-capacity work.
3. Try FP16 only as field-by-field storage ablations with parity.
4. Preserve FP32 arithmetic, reductions, gradients, and moments unless evidence
   supports a narrower change.

## 17. Decision

The kernel fork is successful as a browser systems result:

- the benchmark is now a repeatable Bun command;
- the direct reference remains live;
- parity is active rather than assumed;
- staged backward materially improves 8K and 30K throughput;
- compact projection storage saves memory and modestly improves speed;
- exact separable SSIM improves both low- and higher-resolution steps;
- failed forks and order sensitivity are preserved in the record.

It is not yet a complete research baseline:

- the current representation is still trajectory-gated dynamic 3DGS;
- topology and initialization remain plausible quality bottlenecks;
- the Metal percentage comparison is not matched;
- one scene and one device do not establish generality.

The next high-value move is not another blind shader trick. It is a paired
program: matched wall-time convergence plus phase-guided systems forks. That
keeps speed, quality, memory, and representation claims separately testable.

## 18. Final Verification Addendum

After the profiler-contract patch:

- `npm test` passes all 94 browser tests;
- Node syntax checks pass for the tiled trainer, fast wrapper, kernel lab, and
  Bun runner;
- all consolidated and raw benchmark JSON parses with `jq`;
- `git diff --check` passes;
- a Bun-owned 8K headless smoke launches local Chrome/Dawn, runs both direct
  and staged variants, writes schema-v2 JSON, reports finite matched losses and
  zero overflow, emits `gpuSpanMs` and phase contracts, and leaves no owned
  runner/browser process behind;
- the active Apple parity fixture passes at `1.19e-7` maximum RGB error,
  `5.45e-8` objective error, and 9/9 active gradient families;
- the default SPA smoke reaches step 4,000 at approximately 378 steps/s with
  finite `0.21412` train objective, zero current/cumulative tile overflow,
  4,210 visible splats, 14,092 tile pairs, topology growth, and three updating
  camera panels.

The SPA smoke was paused after inspection. The isolated server remains up.

Operational server backtrack:

- the first live SPA smoke reported message transport because port 8080 was
  owned by a keepalive launchd job running plain `python -m http.server`;
- killing its child process was insufficient because launchd respawned it;
- the stale submitted job `com.openai.codex.dynaworld.browser.8080` was removed
  and resubmitted with `serve_isolated.py`;
- the final response includes COOP `same-origin`, COEP `require-corp`, CORP
  `same-origin`, and `Cache-Control: no-store`, restoring the intended
  `SharedArrayBuffer` transport after refresh.
