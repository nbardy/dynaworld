# Browser WebGPU Contention Validity And Harness Backtracks

## Context

This pass followed the July 31 WebGPU kernel-fork work. The implementation had
already established that staged projected backward was materially faster than
the direct repeated 3D VJP reference, but the user correctly raised a threat to
that conclusion: other jobs on the shared Mac could consume CPU, memory
bandwidth, or Apple GPU time and make runs inconsistent.

The immediate objectives were:

1. make machine contention visible in every new headless artifact;
2. reject unstable measurements rather than average them into a claim;
3. keep experiment output organized;
4. rerun a matched reversed-start comparison under the stronger protocol;
5. preserve failed hypotheses and harness failures as scientific evidence.

Relevant files:

- `web/dynaworld_browser_trainer/benchmarkHostDiagnostics.js`
- `web/dynaworld_browser_trainer/benchmarkStatistics.js`
- `web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js`
- `web/dynaworld_browser_trainer/benchmarkTiledKernels.js`
- `web/dynaworld_browser_trainer/benchmarkDataset.js`
- `web/dynaworld_browser_trainer/dataset.js`
- `web/dynaworld_browser_trainer/benchmark_results/README.md`
- `web/dynaworld_browser_trainer/benchmark_results/runs/2026-07-31/`

## Current Model

Current belief:

> A browser kernel result is promotable only when correctness, per-round
> stability, execution-order controls, and host-environment controls all pass.
> None of these is a substitute for the others.

Confidence: high for the validity contract, medium for the exact contention
thresholds.

The runner now treats the browser page as a compute runtime, not a manual UI.
Bun owns the process lifecycle, starts a private no-store server, launches an
isolated headless Chrome/Dawn runtime, receives the JSON report, captures host
state, writes the artifact, and tears down its owned processes.

The minimal HTML page is still required because WebGPU is exposed through a
browser execution context on this path. It is not required as a visible
benchmark dashboard, and no human interaction is part of the headless
protocol.

## Assumptions

1. The Apple `AGXAccelerator` `Device Utilization %` field is a useful recent
   driver-window contention signal, but not privileged per-process accounting.
2. `ps` CPU percentages are a useful process-pressure proxy, but can be
   transient and are not a direct memory-bandwidth measure.
3. A ten-second delay after owned Chrome teardown is the current conservative
   bound for the benchmark's AGX utilization window to decay on this M4.
4. Alternating both live variants cancels monotonic drift better than two
   separate one-shot processes.
5. A reversed initial order is still needed because alternating four rounds
   does not prove the first initialization/order has no lasting effect.
6. The full-frame kernel lab fixes `motionWeighting=false`; sampled-ray motion
   and static banks are therefore not an input to the measured objective.
7. The current 10% CV threshold is a broad rejection gate, not a claim that a
   9.9% CV benchmark is publication-quality.

## Metrics And Derivations

For measurement round `i`:

```text
s_i = steps in round i
t_i = elapsed wall milliseconds in round i
r_i = 1000 * s_i / t_i                 [steps/s]
mu  = (1 / n) * sum_i r_i
sd  = sqrt((1 / n) * sum_i (r_i - mu)^2)
CV  = sd / mu
```

Throughput is used rather than raw elapsed time because a non-divisible step
count can produce unequal round sizes.

Execution-position bias is:

```text
mu_0 = mean throughput when the variant executes first in a round
mu_1 = mean throughput when the variant executes second in a round
position_bias = abs(mu_1 - mu_0) / ((mu_1 + mu_0) / 2)
```

Host pressure uses dimensionless ratios:

```text
load_per_cpu = load_average_1m / logical_cpu_count

competing_cpu_fraction =
    sum(active_process_cpu_percent) / (100 * logical_cpu_count)

cpu_busy_fraction = user_fraction + system_fraction
```

On macOS/Bun, `os.cpus()` scheduler counters were observed to remain static.
The runner therefore uses the second one-second sample from `top` for CPU busy
and records `cpuBusySource="top-second-sample"`.

The promotion predicate is:

```text
correctness =
    finite_loss
    and zero_tile_overflow

stability =
    at_least_two_rounds
    and round_CV <= configured_max_CV

host =
    preflight_quiet
    and postflight_quiet_after_cooldown

promotable =
    correctness
    and stability_for_every_variant
    and host
```

Current default host thresholds:

| Signal | Threshold |
| --- | ---: |
| CPU busy fraction | <= 0.85 |
| 1-minute load / logical CPU | <= 0.75 |
| aggregate competing CPU / host CPU capacity | <= 0.35 |
| Apple GPU utilization | <= 35% |
| available memory-pressure fraction | >= 0.10 |
| round throughput CV | <= 0.10 |

These thresholds are intentionally permissive enough for local development.
Canonical paper measurements should eventually use tighter, empirically
calibrated bands.

## Observed Chronology

### Observation 1: The machine was initially contended

The first audit observed:

- 1-minute load `5.20` on 10 logical CPUs;
- Steam near `87%` process CPU;
- Codex processes near `70%` and `43%`;
- `mediaanalysisd` near `32%`;
- nontrivial WebTorrent and browser activity;
- 9.68 GiB of 10 GiB swap occupied;
- macOS memory pressure still reported 51% available;
- no thermal or performance warning;
- Apple driver device utilization between `86%` and `96%`;
- latest Apple GPU submitter categorized as `mediaanalysisd`.

Observed fact:

> The GPU was already busy before the benchmark-owned Chrome process existed.

Inference:

> A speed result collected in that window could not establish isolated kernel
> throughput.

Decision:

> Such runs are diagnostic-only. Do not average them with quiet runs.

### Observation 2: The first tiny run timed out

A 512-splat control/candidate smoke entered with preflight Apple GPU
utilization at `69%` and timed out after 180 seconds.

Initial hypothesis:

> External GPU contention caused the timeout.

Status: weakened, then invalidated as the complete explanation.

Why:

A later run started from a quiet GPU and still timed out. The improved timeout
message reported:

```text
state=unset
status=Ready
```

No WGSL initialization or measurement state had begun.

### Observation 3: Sample-bank preprocessing was real overhead, but not the timeout root

The calibrated loader decoded 18 frame atlases and called
`computeMultiviewSamples` across:

```text
17 train cameras * 16 times * 96 * 72 pixels
= 1,880,064 train pixels
```

It created and sorted motion/static candidates even though the full-frame
kernel lab never dispatches the sampled-ray trainer and fixes motion weighting
off.

Action:

The benchmark-only load path now skips this bank and leaves the normal SPA load
semantics unchanged.

Status:

> Valid harness-startup optimization, but not the primary autorun failure.

Backtrack:

After moving the status update before dataset load, the page still showed
`Ready`. That falsified the theory that preprocessing alone blocked the run.

### Observation 4: HTML constraint validation silently blocked autorun

The new CV input used:

```html
min="0.001" step="0.01" value="0.10"
```

Valid values on that lattice are:

```text
0.001 + k * 0.01
```

Therefore `0.100` is invalid; its nearest valid neighbors are `0.091` and
`0.101`. `form.requestSubmit()` performs constraint validation and silently
declined to emit the submit event.

Fix:

```html
min="0.001" step="0.001" value="0.100"
```

Result:

The same tiny headless protocol completed in about 1.3 seconds before expanded
host sampling/cooldown overhead.

This was a benchmark-control-plane bug, not a shader bug.

### Observation 5: Timer granularity rejects tiny workload claims

An 8-step, 256-splat run completed but the direct control had:

- CV `10.33%` in the first tiny smoke;
- CV `26.67%` in a 128-step tiny smoke;
- execution-position relative difference near `33%` in the latter.

The candidate was more stable, but this does not make the comparison
promotable. At this workload, per-round times were only a few milliseconds and
warm-state/timer quantization dominated.

Decision:

> Use 8K as the minimum current headline workload for this kernel pair.

### Observation 6: Bun CPU counters failed silently

The first successful artifact serialized `cpuBusyFraction=null`.

Direct test:

- Node's `os.cpus()` counters advanced over 250 ms.
- Bun's counters were unchanged over the same interval.

Fix:

Use two macOS `top` samples one second apart and parse the second CPU line.
Keep `os.cpus()` deltas as a non-macOS/fallback source. Missing Apple GPU
telemetry on macOS fails closed for promotion.

### Observation 7: AGX postflight needs a decay window

An immediate postflight showed `94–96%` because the driver window still
contained the benchmark's final work.

One-second cooldown:

- still observed `95%`;
- fell to `2%` a couple of seconds later.

Three-second cooldown:

- one run fell to `3%`;
- another still showed `94%`, with WebTorrent as the latest submitter;
- fell to `3%` about two seconds later.

Five-second cooldown on 128-step workloads:

- both final canonical runs measured `0–3%` postflight GPU;
- both passed the complete host gate.

Five-second cooldown on a 512-step 192x144 SSIM workload:

- one postflight still measured `80%` despite a `0%` preflight and quiet
  CPU/load/process checks;
- this falsified the five-second default as a general bound.

Current decision:

> Keep the postflight GPU gate and use a fixed ten-second decay window for new
> runs on this M4. Do not retry until a result happens to look quiet, and do
> not simply ignore high postflight utilization.

## Canonical V3 Results

Workload:

- Apple M4;
- 8,192 active and capacity splats;
- 96x72 full-frame raster;
- 32 warmup steps;
- 128 measured steps in four alternating rounds;
- five timestamped GPU profiles per variant;
- direct repeated 3D VJP control;
- staged projected-gradient candidate;
- packed FP16 checkpoints;
- exact separable 11x11 SSIM;
- finite loss and zero tile overflow.

Artifacts:

- `web/dynaworld_browser_trainer/benchmark_results/runs/2026-07-31/backward_8k_control_first_v3_apple_m4.json`
- `web/dynaworld_browser_trainer/benchmark_results/runs/2026-07-31/backward_8k_candidate_first_v3_apple_m4.json`

| Metric | Control first | Candidate first |
| --- | ---: | ---: |
| direct throughput | 717.89 steps/s | 719.10 steps/s |
| staged throughput | 1242.72 steps/s | 1259.84 steps/s |
| staged wall speedup | 1.7311x | 1.7520x |
| direct GPU-span p50 | 1.4093 ms | 1.3385 ms |
| staged GPU-span p50 | 0.7850 ms | 0.7763 ms |
| staged GPU-span speedup | 1.7954x | 1.7243x |
| maximum round CV | 0.60% | 1.00% |
| maximum position bias | 0.39% | 0.45% |
| preflight GPU | 2% | 0% |
| postflight GPU | 3% | 0% |
| promotable | yes | yes |

Host ranges across final artifacts:

- CPU busy: `14.4–24.1%`;
- load per logical CPU: `0.243–0.331`;
- process CPU fraction: `0.107–0.190`;
- Apple GPU: `0–3%` at accepted boundaries;
- no memory-pressure or thermal warnings.

The prior `v2` 8K result reported `1.84x` wall and `1.80x` GPU-step speedups.
The `v3` wall mean is about `1.742x`, around 5% lower than that historical
single summary, while the GPU-span mean is about `1.760x`. This is not evidence
of a shader regression:

- both new initial orders agree closely;
- direct and staged throughputs are stable;
- host validity is explicit;
- GPU-span semantics differ from the older sum-of-phase headline;
- the older artifact lacks the new host controls.

The `v3` pair is stronger evidence and should be the current systems baseline.

## Expansion Pass 2: Scaling And Pair-Level Reproducibility

The first protocol version made each run internally auditable but still left a
human to decide whether two reversed-start runs agreed. That is insufficient
for small effects.

For a metric `q` measured in runs `a` and `b`, pair drift is:

```text
pair_drift(q) = abs(q_a - q_b) / ((abs(q_a) + abs(q_b)) / 2)
```

The pair summary now checks:

```text
wall speedup drift <= 0.05
GPU speedup drift <= 0.10
each variant's absolute throughput drift <= 0.05
```

It also verifies identical workload options, dataset dimensions, result IDs,
opposite start orders, and individually promotable source artifacts.

### 30K backward

The first `30K` v3 run used the runner's 1,024-entry tile capacity.

Observed:

- host and round stability passed;
- direct overflow was `387,851` tile/splat pairs;
- staged overflow was `387,745`;
- the apparent wall speedup was `1.975x`.

Decision:

> Reject the result. The apparent nearly 2x gain is partly a different,
> truncated workload.

Rerun:

- 4,096-entry tile capacity;
- zero overflow;
- control-first wall/GPU speedup `1.8181x / 1.7409x`;
- candidate-first wall/GPU speedup `1.8177x / 1.7514x`;
- pair wall-speedup drift `0.026%`;
- maximum variant-throughput drift `0.282%`;
- pair promotable.

Interpretation:

The staged design's advantage grows at 30K, and the pair is more reproducible
than the 8K pair. This supports the original theory that eliminating repeated
3D VJPs matters more as projected pair count grows.

### 30K compact projection

Results:

- wall speedup `1.0988x` and `1.0838x`;
- GPU-span speedup `1.1033x` and `1.0865x`;
- wall-speedup drift `1.38%`;
- maximum variant-throughput drift `1.15%`;
- both orders zero overflow and pair promotable.

Interpretation:

The compact packet's gain is real but modest. It remains worth keeping because
it also saves 3.84 MB at 30K relative to the monolithic packet, but it should
not be marketed as a large algorithmic win.

### 192x144 separable SSIM

The first 128-step pair produced:

- wall speedup `1.252x` and `1.366x`;
- one naive run at `9.7%` CV;
- execution-position bias up to `11.2%`.

Although each run barely cleared the loose 10% CV threshold, the pair did not
deserve a precise claim.

Rerun:

- 64 warmup steps;
- 512 measured steps;
- nine GPU profiles;
- fixed ten-second postflight cooldown;
- wall speedup `1.2876x` and `1.2829x`;
- GPU-span speedup `1.2800x` and `1.2923x`;
- wall-speedup drift `0.36%`;
- maximum variant-throughput drift `0.34%`;
- pair promotable.

Interpretation:

The separable exact 11x11 SSIM improvement survives the stronger protocol at
about `1.285x` wall throughput. The longer run changed confidence far more than
it changed the central estimate.

### Pair artifacts

- `backward_8k_pair_v3_apple_m4.json`
- `backward_30k_pair_v3_apple_m4.json`
- `projection_30k_pair_v3_apple_m4.json`
- `ssim_8k_192x144_pair_v3_apple_m4.json`

All live under:

```text
web/dynaworld_browser_trainer/benchmark_results/runs/2026-07-31/
```

## Evidence Classification

Observed:

- process/load/memory/AGX snapshots saved in artifacts;
- exact per-round times and throughputs;
- both order directions;
- finite losses and overflow counts;
- timestamped phase/GPU spans;
- timeout state/status;
- Bun vs Node CPU-counter behavior.

Inferred:

- media analysis was a material initial GPU competitor;
- tiny-workload CV was dominated by timing/warm-state effects;
- the ten-second window normally removes owned benchmark utilization from the
  AGX postflight.

Speculative:

- swap occupancy contributed meaningfully to any particular timing;
- WebTorrent caused the rejected three-second postflight rather than merely
  becoming the latest small submitter;
- a 35% AGX threshold is optimal across all Apple chips and OS versions.

Proposed:

- calibrate thresholds from repeated idle/loaded distributions;
- run the accepted pairs on multiple quiet occasions;
- retain only promotable artifacts as canonical evidence.

## Alternative Branches

### Branch A: Constant-rate external GPU work remains invisible

Hypothesis:

An external workload begins after preflight, remains constant, and does not
raise round CV.

Why it might be true:

Pre/post snapshots do not continuously attribute GPU work.

What would make it false:

Privileged per-process GPU accounting or a continuous independent GPU sampler.

Cheap test:

Repeat the same pair three times. Compare absolute direct/staged p50 GPU spans,
not only their ratio.

If supported:

Add a continuous sidecar sampler or require a tighter multi-repeat absolute
timing band.

If invalidated:

Keep current snapshots plus repeated orders.

### Branch B: Ten-second cooldown is machine-specific

Hypothesis:

Other Apple chips or OS versions retain AGX utilization longer or shorter.

Cheap test:

Record utilization at 0, 1, 3, 5, and 8 seconds after a fixed queue workload.

If supported:

Replace the fixed delay with bounded polling for two consecutive quiet samples.

If invalidated:

Keep the simple fixed delay, revisiting its value when workloads or hardware
change.

### Branch C: Skipping sampled-ray banks accidentally changes a hidden input

Hypothesis:

An inherited buffer or target alpha path affects the full-frame tiled
objective despite `motionWeighting=false`.

What would make it false:

One-step image/objective/gradient parity between loaders with samples enabled
and disabled.

Current code evidence:

The tiled shader selects weight `1.0` when `motionWeighting==0`; sampled-ray
indices are not dispatched by the tiled step.

Cheap test:

Add a live one-step A/B harness if the kernel lab later enables motion
weighting.

Decision:

The optimization is accepted for the current fixed benchmark options. It must
be revisited if motion weighting becomes a benchmark axis.

### Branch D: CV alone misses deterministic order bias

Hypothesis:

Each variant has low internal CV but is consistently faster in the second
execution position.

Current mitigation:

Execution-position means and relative difference are recorded separately;
both initial orders are run.

Future gate:

Consider an explicit position-bias threshold after collecting enough valid
artifacts to choose it empirically.

### Branch E: The direct control is no longer an honest reference

Hypothesis:

Shared setup, projection layout, or SSIM changes have drifted between control
and candidate.

What would make it false:

Active one-step parity, identical options outside the explicit variant
override, and matching loss trajectories.

Current evidence:

The existing live parity gate reports maximum RGB error `1.19e-7`, objective
error `5.45e-8`, accepted gradients in all nine active families, and zero
overflow. Final v3 loss deltas are around `3.4e-4` after many interleaved
optimizer steps, consistent with different floating-point reduction/update
order rather than a different objective.

## Log Hygiene Decision

Durable:

- eight final promotable `v3` run artifacts and four pair summaries;
- browser README summary;
- benchmark-results contract README;
- this append-only loose note;
- focused unit tests.

Scratch only:

- initial 69–96% GPU-contended attempts;
- two invalid-form timeout runs;
- 8-step and 256-splat CV smokes;
- the overflowing 30K/1,024-capacity run;
- the short unstable SSIM pair;
- intermediate one-, three-, and five-second cooldown artifacts.

Reason:

Keeping every failed JSON beside canonical results makes later agents infer a
standings table from file presence. Failed runs remain described here with
their falsification value, while raw scratch files stay in `/tmp`.

## Next Falsification Lanes

1. Repeat the accepted 8K and 30K pairs on three quiet occasions and build an
   absolute timing distribution.
2. Measure `30K, 192x144` only after sizing the tile-capacity/memory envelope
   and proving zero overflow.
3. Add a tile-occupancy histogram so high-percentile pressure chooses capacity
   before a benchmark, rather than discovering overflow afterward.
4. Add bounded postflight polling only if another machine shows a different
   AGX decay window.
5. If motion weighting becomes an axis, restore sample-bank construction and
   add loader parity before timing it.
6. Keep browser systems results out of `BASELINES.md` until they satisfy the
   project data/eval contract; these are kernel-system artifacts, not paper
   quality rows.

## Final Decision

The staged backward result survives a substantially stronger protocol. The
accepted claim is:

> On the matched 8K/96x72 Apple M4 full-frame browser workload, staged
> projected backward is about 1.73–1.75x faster in end-to-end measured
> throughput than the direct repeated 3D VJP reference, with both reversed
> initial orders promotable under explicit host, stability, correctness, and
> postflight gates.

The accepted scaling claims are:

- staged backward is `1.818x` faster by wall throughput at 30K;
- compact projection is `1.084–1.099x` faster at 30K while using less memory;
- exact separable SSIM is `1.283–1.288x` faster at 192x144;
- each claim passes individual-run and reversed-start pair gates.

The rejected claim is:

> Any number emitted while the shared machine is busy is good enough if the
> candidate/control ratio looks plausible.

The process-control work was not incidental. It uncovered two control-plane
bugs, prevented contended measurements from entering the record, and converted
the speedup from a plausible local observation into a reproducible systems
result with named limitations.

## Final Acceptance Gate

After the final fail-closed telemetry hardening:

- `npm test` passed all 111 browser tests;
- all 12 durable JSON files parsed successfully;
- all four reversed-start pair summaries remained promotable;
- a deliberately duplicated control-first pair was written as
  `diagnostic-only` and the summarizer exited with status `2`;
- `git diff --check` passed;
- the launchd-owned isolated server remained healthy with COOP, COEP, CORP,
  and `Cache-Control: no-store` headers.

This final gate matters because invalid-pair rejection and missing-telemetry
rejection are automation contracts. A warning rendered in JSON would not be
enough if a sweep process still received a successful exit code.
