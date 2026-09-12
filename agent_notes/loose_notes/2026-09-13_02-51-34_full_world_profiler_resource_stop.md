# Full-world operation profiling exceeded the process-memory budget

The previous turn was measured progress: STAR 68fa855/root 6bbcccd preserved
F4/F3 numerical checks and reduced warm forward/backward 3.97 -> 1.02 s.
This continuation attempted operation-level attribution of the remaining
backward cost. It produced a resource-stop result, not timing or operator
evidence. No renderer or acceptance threshold changed.

## Attempt and authoritative stop

The current source/index state was checked, and a live process inventory
confirmed no accelerator job remained. The existing guarded launcher passed
its live host preflight. The diagnostic used the same 256-tube saved world,
four frames and two-frame resident chunks as the preceding benchmarks.
It performed one uninstrumented render/backward warmup and requested one
CPU-activity Kineto profile of render/backward with record_shapes=True,
profile_memory=False and with_stack=False. It did not request GPU activity
timing, so even a completed result would have described CPU dispatch cost.

Shell session 15523 returned exit 1. Its retained log shows the independent
LocalResourceMonitor raised:

    RuntimeError: local playground exceeded process-tree RSS limit

The unchanged limit is 3,221,225,472 bytes (3 GiB), counting the child tree
and launcher. The launcher finally block terminates the job's process group
and waits, escalating to SIGKILL if necessary. A subsequent live inventory
confirmed the profiler process was gone and no accelerator job remained.
The exact observed peak is unavailable: the exception interrupts normal
receipt serialization. Do not invent a peak or infer that the host itself
ran out of memory. No completed operator_profile.json or resource receipt
exists for this failed attempt. There is no usable operator ranking.

## Interpretation and scope of the stop

Installed torch/profiler/profiler.py lines 141-144 explicitly document that
shape recording holds tensor references and may introduce additional copies.
That makes instrumentation overhead a plausible explanation; this attempt
does not isolate its contribution causally. In the prior ordinary guarded
benchmark/profile bundle, maximum tree/launcher RSS was 1,822,441,472 bytes.
Do not turn the instrumented stop into a claim that ordinary rendering has
regressed or now exceeds its memory budget.

The full-world shape-recording profiling lane is stopped. It was not retried
with a larger cap, a renamed full-world profiler, or another concurrent job.
The existing source snapshots and pstats remain useful but cannot determine
which GPU backward operation dominates. Future attribution must account for
instrumentation memory within the same resource contract; the missing operator
ranking is unresolved. The preceding batched-fallback numerical and repeated
timing results remain the current accepted local implementation evidence.

Raw script, launcher, preflight, source hashes and failure log are retained at
`outputs/benchmarks/2026-09-13_fallback_backward_ops/before/`. All bound source
hashes were rechecked after the stop and still match. W&B was omitted because
this was a mechanical profiler diagnostic. No benchmark, training run, native
build, upload, application termination, or source refactor followed the trip.

This is the first profiling-resource blocker after the prior successful turn;
the broad goal is still active, not complete or globally blocked. Other known
work remains, but do not re-run this failed profiling mode automatically.
Public paper counts remain 0/7 contexts and 0/21 lanes. BASELINES is unchanged.
