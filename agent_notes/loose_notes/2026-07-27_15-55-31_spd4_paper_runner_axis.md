# SPD(4) paper-runner representation axis

## Context

The live STAR comparison trainer now exposes:

```text
--uvt-world-representation {legacy_tube,full_spd4}
```

but `research_experiments/paper_runner_suite/run_unified_paper_ablation.py`
could only launch the historical restricted tube. This session added the same
axis to the unified runner while retaining `legacy_tube` as the default.

Files inspected:

- `research_experiments/paper_runner_suite/run_unified_paper_ablation.py`
- `third_party/fast-mac-gsplat/variants/star_uvt_v0/research_project/benchmarks/multicam_heldout_compare.py`
- `tests/test_unified_paper_ablation.py`

## Current model

Observed fact:

- The comparison trainer records `uvt_world_representation` in report metadata
  and accepts exactly `legacy_tube` and `full_spd4`.
- Its full-SPD(4) compiler currently supports only `static_view`.

Implementation decision:

- Thread one normalized string through command construction, isolated child
  processes, dry-run manifests, execution, final summaries, and W&B identity.
- Interpret a missing report metadata field as `legacy_tube`. This is the
  backward-compatibility rule for reports written before the axis existed.
- Fail when the requested representation and report metadata disagree. This
  prevents `--reuse-existing` from silently relabeling a legacy artifact as an
  SPD(4) result.
- Reject `full_spd4` at command construction when the protocol selects a
  segmented/moving-camera chart. In particular, the current D-NeRF fallback
  remains legacy-only.

## Assumptions and boundaries

- The SPD(4) atom lowers into the mature STAR UVT rasterizer, so the kernel
  registry remains `family=star_uvt`; the producer representation is recorded
  separately.
- Both isolated comparison child processes receive the representation flag.
  The dynamic-3DGS-only child does not train World Tubes, but recording the same
  flag in both reports makes cross-child metadata matching explicit.
- Output-directory selection is still the caller's responsibility. Running
  both representations concurrently requires distinct `--out-dir` values.
- The default legacy W&B run hash is preserved. A non-default SPD(4) run adds
  the representation to its hash/name, preventing it from resuming into the
  legacy W&B run.

## Falsification tests

1. Command construction:
   - Default command must emit `legacy_tube`.
   - Explicit SPD(4) must reach both isolated child commands.
2. Artifact integrity:
   - Merging lane reports with different representation metadata must fail.
   - Two old reports with no representation field must merge as legacy.
   - Requesting SPD(4) while validating a legacy report must fail.
3. Camera boundary:
   - Requesting SPD(4) for the segmented D-NeRF protocol must fail before
     launching a child process.
4. Actual CLI:
   - A static-view smoke protocol dry run with `full_spd4` must serialize the
     axis at the top level and in both child commands.

Results:

```text
PYTHONPATH=src/train /opt/homebrew/bin/pytest \
  tests/test_unified_paper_ablation.py -q

19 passed in 1.34s
```

The actual CLI dry run also completed on CPU and emitted
`--uvt-world-representation full_spd4` for both isolated comparison children.
No training or MPS workload was launched.

## Red-team branches

Hypothesis:
    A reused merged report can still be stale.
Why it might be true:
    `materialize_isolated_comparison_report` returns an existing merged path
    directly when reuse is enabled.
Current mitigation:
    `execute` immediately validates the merged report against the requested
    representation before logging or constructing the final summary.
Cheap test:
    Seed a legacy merged report, request SPD(4), and call `execute` with external
    effects mocked.
If invalidated:
    Move representation validation into the materializer's early-return path.

Hypothesis:
    Static-view SPD(4) is enough for the initial matched coffee/martini row.
What would make it false:
    A paper protocol or future atlas lane requires moving-camera compilation.
Decision implication:
    Do not relax the early failure until the SPD(4) camera-atlas compiler has a
    tested implementation.

## Next actions

- Run the matched legacy/SPD(4) smoke or capacity fixture with separate output
  directories.
- Add moving-camera/atlas lowering before enabling `full_spd4` on D-NeRF.
- If full paper rows are scheduled, keep the existing local-MPS safety gate;
  this axis does not make a paper-scale workload safe.
