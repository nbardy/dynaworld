# STAR UVT V-JEPA Gaussian Comparison

## Goal

Repeat the core STAR UVT feature-shader plan in the routing docs, fill any
missing details, execute the next benchmark/report gates, and record progress.

## What Changed

- Added `research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py`.
- Regenerated the all-renderer matrix with individual first-class STAR JSON
  ingestion:
  - `outputs/benchmarks/2026-05-19_renderer_scaling_report.md`
  - `outputs/benchmarks/2026-05-19_renderer_scaling_report.csv`
- Generated the normalized STAR/Gaussian V-JEPA comparison:
  - `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.md`
  - `outputs/benchmarks/2026-05-19_star_uvt_vjepa_vs_gaussian_comparison.json`
- Updated routing docs:
  - `README.md`
  - `PROJECT_INDEX.md`
  - `TODO/README.md`
  - `EXPERIMENTS.md`
  - `research_experiments/star_uvt_feature_tubes/README.md`
  - `research_experiments/star_uvt_feature_tubes/2026-05-18_fast_shader_port_plan.md`
  - `agent_notes/key_learnings.md`

## Core Current State

- `star-feature-512-fast` is still the selected RGB-target speed diagnostic,
  not a precomputed V-JEPA route.
- The real STAR V-JEPA target route is separate and now has a chunked
  64f/512px/8192t/F32 scale gate.
- The 300-clip Gaussian/token route remains the dataset-scale route, but its
  512px multires promotion is blocked by NaNs and its differentiable
  prediction-side V-JEPA loss is still too slow.

## New Comparison Numbers

Matched 64f/512px/8192 rows from the normalized report:

| route | step | backward | render | target/loss |
| --- | ---: | ---: | ---: | ---: |
| STAR V-JEPA chunked target | 3.743s | 1.077s | 0.816s | 1.734s |
| STAR RGB fast feature diagnostic | 2.491s | 1.184s | 0.911s | 0.287s |
| Gaussian/token recon-only cached conditioning | 3.460s | 1.963s | 0.274s | 0.002s |
| Gaussian/token prediction-side V-JEPA loss | 38.621s | 36.762s | 0.213s | 0.617s |

Interpretation:

- STAR V-JEPA is no longer blocked on "does it use cached features?" It does.
- STAR V-JEPA is not currently rasterizer-only limited; target interpolation
  and target loss are the largest bucket.
- The old 36-39s Gaussian V-JEPA run is still explained by frozen V-JEPA in the
  prediction backward path, not by cached feature loading.
- Multicam cached-V-JEPA 16f/128px rows are useful references but not matched
  64f/512px evidence.

## Renderer Matrix Update

`research_experiments/renderer_scaling_report.py` now ingests standalone
first-class STAR feature JSONs, not only the older scale summary JSON. The
refreshed report has 145 rows and includes the selected no-pre-norm
`feature_direct_gradcache_reduce_vec4` rows.

The report remains a mixed matrix:

- STAR RGB direct-kernel rows are synthetic kernel probes.
- Dynamic RGB and projected F32 rows are projected-raster synthetic rows.
- STAR F32 first-class rows are real-video trainer steps.
- 2-step rows are timing smokes; use the selected-shader report for keeper
  decisions.

## Remaining Plan

1. Cache or fuse the STAR adapted V-JEPA target layout so the `1.734s` target
   chunk/loss bucket does not dominate the V-JEPA target route.
2. Port a true scalar fixedbin/tile-slot feature-gradient accumulator or native
   image-space VJP/handoff. Do not spend the next gate on duplicate traversal
   two-pass variants.
3. Fix or guard Gaussian/token 512px promotion NaNs before treating the 300-set
   multires route as a completed scale baseline.
4. Keep feature STAR quality work focused on objective/decoder shape because
   Gate 4 still loses badly to RGB STAR source overfit.

## Validation

Passed:

- `rtk .venv/bin/python -m py_compile research_experiments/renderer_scaling_report.py research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_bridge_audit.py src/train/train_star_uvt_feature_overfit.py`
- `rtk .venv/bin/python research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py`
- `rtk .venv/bin/python research_experiments/renderer_scaling_report.py --out-md outputs/benchmarks/2026-05-19_renderer_scaling_report.md --out-csv outputs/benchmarks/2026-05-19_renderer_scaling_report.csv --report-date 2026-05-19`
- `rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_feature_target_adapter.py -q`
- JSON sanity checks for the comparison report and V-JEPA bridge audit.
- `wc -l agent_notes/key_learnings.md` -> `195`.
- `git diff --check`.
- `git -C third_party/fast-mac-gsplat diff --check`.

No active `train.py`, `train_star_uvt_feature_overfit`, bridge-audit, or
comparison process was found after validation.
