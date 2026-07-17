# STAR Report JSON Loader Cleanup

## Context

After the STAR report helper grew `load_report_json(...)`, several report
scripts still loaded report-shaped JSON objects with local `json.load(...)` or
`json.loads(path.read_text(...))` snippets. These were not different contracts:
they expected a JSON object and should fail if the file is malformed or has the
wrong top-level shape.

One nearby loader was intentionally left alone:
`star_uvt_vjepa_bridge_audit.py` returns `None` for missing files and
`{"_load_error": ...}` for failed loads, so it is a tolerant audit surface, not
the strict report-object loader contract.

## Changes

- Routed `target_cache_budget.py` through `load_report_json(...)` for the
  observed cached-chunks and target-grid result inputs.
- Routed `firstclass_scale_report.py` result JSON inputs through
  `load_report_json(...)` and removed its local `_load_json(...)` helper.
- Routed `star_uvt_vjepa_vs_gaussian_comparison.py` report loads through
  `load_report_json(...)` and removed its local `_load_json(...)` helper.
- Routed `star_uvt_feature1_wholegraph_profile.py` reference timing loads
  through `load_report_json(...)`.
- Updated `TODO/trainer_landscape_unification.md` and `CODE_ORGANIZATION.md`.

## Validation

```bash
PYTHONPATH=src/train:. uv run python -m py_compile \
  research_experiments/star_uvt_feature_tubes/report_artifacts.py \
  research_experiments/star_uvt_feature_tubes/target_cache_budget.py \
  research_experiments/star_uvt_feature_tubes/firstclass_scale_report.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_vjepa_vs_gaussian_comparison.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  tests/test_star_uvt_report_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_star_uvt_report_artifacts.py -q
```

Passed: `13 passed in 0.31s`.

`rg` no longer finds local strict JSON object loaders or direct
`json.loads(path.read_text(...))` usage in the routed scripts. The known parent
`uv` warning about `gsplats_browser/pyproject.toml` lacking `[project]` remained
but the commands exited `0`.
