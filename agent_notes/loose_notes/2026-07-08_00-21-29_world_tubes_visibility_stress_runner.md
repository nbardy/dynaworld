# World Tubes visibility stress runner

## Context

The active goal is to build full runner coverage so the World Tubes and
WorldFoam papers can ablate and compare representations. The previous chunk
added the first decisive-demo fixture runner. This chunk added the next World
Tubes runner: visibility stress accounting.

## Implemented

Added:

- `research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py`
- `tests/test_star_uvt_projective_visibility_stress_suite.py`

The runner writes:

- `outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json`
- `outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.md`

The suite currently has four CPU fixture rows:

- `clean_orbit_ordered`: stable compiled interval row, noncollapsed.
- `crossing_raw_interval`: raw depth-order crossing, visibility-stale,
  collapsed by `quality_error_without_fallback`.
- `crossing_stratified`: visibility-stratified repair, noncollapsed.
- `forced_fallback_ambiguous`: forced ambiguous fallback row, collapsed by
  fallback cell/sample fractions.

The fixture records both deterministic interval-work `runtime_ratio` and
measured CPU `measured_runtime_ratio`. Collapse logic uses the deterministic
ratio so tiny CPU timing noise cannot falsely collapse a repaired row.

## Verification

Passed:

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_projective_visibility_stress_suite.py -q
```

Result:

```text
7 passed in 3.17s
```

Passed:

```bash
PYTHONPATH=src/train uv run python -m py_compile research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py
```

Passed:

```bash
PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py \
  --out-dir outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite

PYTHONPATH=src/train uv run python research_experiments/star_uvt_feature_tubes/projective_visibility_stress_suite.py \
  --verify-report outputs/benchmarks/2026-07-08_star_uvt_projective_visibility_stress_suite/summary.json
```

Saved summary:

- collapsed cases: `crossing_raw_interval`, `forced_fallback_ambiguous`
- noncollapsed cases: `clean_orbit_ordered`, `crossing_stratified`
- `max_quality_error=0.1867423951625824`
- `max_fallback_sample_fraction=1.0`

## Important boundary

This is a paper-runner fixture and verifier, not yet a real-video stress
benchmark. It proves that the report can expose raw crossing failure, repaired
stratification, and fallback dominance instead of hiding them.

## Next work

1. Extend `projective_decisive_demo_report.py` with real-video/media rows:
   contact sheet, fallback heatmap, runtime bars, memory bars, and saved
   artifact verification.
2. Add the WorldFoam cell-path optical-transfer fixture runner from
   `research_notes/worldfoam_paper/experiment_designs/cell_path_optical_transfer_fixture.md`.
3. Add the WorldFoam owner-run VJP/finite-difference runner.
4. Add a shared paper table/chart generator that consumes saved World Tubes,
   WorldFoam, and dynamic 3DGS reports.
