# World Tubes decisive-demo runner slice

## Context

The user asked what work is needed to implement full runners so the World
Tubes and WorldFoam papers can ablate and compare the approaches. I treated
this as implementation work for the first paper-runner spine, not just a plan.

The repo already had a code-level plan for a World Tubes decisive demo, but the
named file did not exist yet:

- `research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py`
- `tests/test_star_uvt_projective_decisive_demo_report.py`

## Implemented

Added `research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py`.

The runner builds a tiny two-trace projective cell atlas in two routes:

- `per_frame_replay`: one cell per frame.
- `compiled_interval_atlas`: one interval cell spanning all frames.

It renders both through the existing
`render_projective_trace_cell_atlas_reference(...)` path, compares the compiled
route against per-frame replay, and writes:

- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.md`

The verifier checks:

- benchmark/mode shape
- replay and compiled rows present
- interval entries and dense samples are consistent
- compiled atlas compresses dense samples
- fallback cell/sample fractions stay zero
- image error and PSNR stay inside thresholds
- stale summaries are rejected
- media-mode reports include required artifact paths

Added `tests/test_star_uvt_projective_decisive_demo_report.py` with CPU-only
contract tests for the valid fixture, stale summary, hidden fallback, quality
regression, missing replay route, missing media artifacts, and optional saved
artifact validation.

Updated:

- `TODO/README.md`
- `EXPERIMENTS.md`

## Verification

Passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_decisive_demo_report.py -q
```

Result:

```text
7 passed in 4.77s
```

Passed:

```bash
PYTHONPATH=src/train uv run python -m py_compile research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py
```

Passed:

```bash
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --fixture-only \
  --out-dir outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture

PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_decisive_demo_report.py \
  --verify-report outputs/benchmarks/2026-07-08_star_uvt_projective_decisive_demo_fixture/summary.json
```

The saved report has:

- `max_image_abs_error_vs_reference=0.0`
- `min_psnr_vs_reference=120.0`
- `compiled_to_replay_interval_entry_ratio=0.125`
- `compiled_to_replay_memory_ratio=0.216`

## Important boundary

This is a correct first runner spine, not a paper-quality real-video result.
It proves replay equivalence and report/verifier mechanics on a small fixture.
It does not prove real-video quality, heldout performance, or native Metal
throughput.

## Next work

1. Add `projective_visibility_stress_suite.py` with crossing-depth,
   finite-exposure, rolling-shutter, fallback-strata, and failure rows.
2. Extend `projective_decisive_demo_report.py` with real-video/media rows:
   contact sheet, fallback heatmap, runtime bars, memory bars, and saved
   artifact verification.
3. Add the WorldFoam cell-path optical-transfer fixture runner from
   `research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`.
4. Add the WorldFoam owner-run VJP/finite-difference runner.
5. Add a shared paper table/chart generator that consumes saved World Tubes,
   WorldFoam, and dynamic 3DGS reports.
