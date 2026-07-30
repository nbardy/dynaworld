# Browser WebGPU Benchmark Artifacts

This directory contains durable benchmark evidence for the browser trainer.
It is not a scratch directory.

## Layout

- Root-level dated JSON files are curated summaries or historical canonical
  runs.
- `2026-07-31_interleaved/` contains the raw alternating-order runs behind the
  July 31 kernel-fork summary.
- `runs/YYYY-MM-DD/` is the destination for new auto-named `v3` artifacts.
- `runs/2026-07-31/*_pair_v3_apple_m4.json` contains the
  contention-qualified, reversed-start canonical pair summaries.
- One-step, timeout, parser, and CLI smokes belong under `/tmp`, not here.

Create an orderly run with:

```bash
bun web/dynaworld_browser_trainer/run_headless_kernel_benchmark.js \
  --experiment backward --variant both --splats 8192 \
  --warmup 32 --steps 128 --profiles 5 \
  --contention-policy fail \
  --out-dir web/dynaworld_browser_trainer/benchmark_results/runs \
  --run-id staged-backward-8k-control-first
```

The runner writes using the host-local date/time (the JSON `recordedAt` remains
UTC and `artifact.filenameTimeZone` records the filename zone):

```text
runs/YYYY-MM-DD/HH-MM-SS_<sanitized-run-id>.json
```

Use a run id that names the changed axis, workload, and initial order. Matched
repeats should normally end in `control-first` and `candidate-first`.

## Promotion Contract

A `v3` artifact is headline evidence only when
`validity.promotable=true`. That requires:

1. finite loss for every measured variant;
2. zero tile overflow;
3. at least two timed rounds per variant;
4. throughput CV no greater than the configured `maxRoundCv`;
5. quiet preflight and postflight host diagnostics;
6. an apples-to-apples workload and objective;
7. a reversed-start repeat before changing a default;
8. pair drift no greater than 5% for wall speedup, 10% for GPU speedup, and
   5% for either variant's absolute throughput.

`warn` and `record` policies do not waive these gates. They only control
whether contention is printed to stderr. A contended artifact remains useful
for debugging and is labeled diagnostic-only.

Historical `v1`/`v2` artifacts predate host diagnostics and round-CV validity.
They remain provenance for past decisions, but should not be silently promoted
to the `v3` evidence standard.

Build the pair gate with:

```bash
bun web/dynaworld_browser_trainer/summarize_headless_kernel_pair.js \
  path/to/control-first.json path/to/candidate-first.json \
  --out path/to/pair-summary.json
```

The pair tool also rejects mismatched workloads, duplicated start order, or an
individually non-promotable source artifact.

## Contention Limits

The runner records sanitized process basenames/categories, host CPU/load/memory
pressure, macOS thermal status, and an Apple GPU-driver snapshot. It never
stores process arguments or PIDs.

These checks cannot provide privileged per-process GPU accounting. A quiet
pre/post snapshot can miss mid-run work, and a constant-rate competitor can
leave round CV low. Postflight waits ten seconds after owned Chrome teardown so
the benchmark's own driver window can decay before GPU validity is evaluated.
Alternating variants, reversed-start repeats, timestamped GPU spans, and
explicit host diagnostics are complementary controls rather than
interchangeable proofs.
