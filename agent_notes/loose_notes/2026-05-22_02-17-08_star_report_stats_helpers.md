# STAR Report Stats Helpers

## Context

The STAR UVT feature-tube report/profiling scripts still had several local
`_stats(...)` helpers after the earlier report-artifact cleanup. A live check
showed there were two real contracts, not one:

- zero-empty `{samples, mean, min, max}` summaries used for timing phase samples
- count/stdev distributions with `None` empty values used for repeat timing
  distributions

Keeping those shapes explicit matters because downstream JSON/markdown reports
expect different empty-value semantics.

## Changes

- Added `summary_stats(...)` to
  `research_experiments/star_uvt_feature_tubes/report_artifacts.py`.
- Added `distribution_stats(...)` to the same helper module.
- Routed the zero-empty summary contract through `summary_stats(...)` in:
  - `star_uvt_logit_handoff_rgb_vjp_profile.py`
  - `star_uvt_feature1_wholegraph_profile.py`
  - `sparse_visual_loss_vjp_profile.py`
- Routed the count/stdev distribution contract through `distribution_stats(...)`
  in:
  - `sparse_forward_batched_target_vjp_profile.py`
  - `sparse_forward_batched_step_benchmark.py`
  - `sparse_forward_timing_repeat.py`
- Added focused helper tests in `tests/test_star_uvt_report_artifacts.py`.
- Routed `tile_slot_accumulator_budget.py` through
  `report_artifacts.split_csv_ints(...)` instead of its local `_split_csv_ints`
  copy.
- Updated `TODO/trainer_landscape_unification.md` and `CODE_ORGANIZATION.md`.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_target_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_batched_step_benchmark.py \
  research_experiments/star_uvt_feature_tubes/sparse_forward_timing_repeat.py \
  research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  tests/test_star_uvt_report_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_star_uvt_report_artifacts.py -q
```

Passed: `13 passed in 0.14s`.

After routing the tile-slot parser, the focused compile and report-artifact test
were repeated:

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/tile_slot_accumulator_budget.py \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  tests/test_star_uvt_report_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_star_uvt_report_artifacts.py -q
```

Passed: `13 passed in 0.13s`.

`uv` still emits the known parent-project warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` lacking a
`[project]` table; the commands exited `0`.
