# STAR UVT Colorizer Factory Boundary

Date: 2026-05-21 21:16:12 +07

## Goal

Continue the trainer modularization pass by removing duplicated
config-to-`FeatureToColor` construction from STAR UVT diagnostic and profiling
scripts. The shared factory already existed for the feature-overfit trainer and
RGB probes, but several diagnostics still copied the same kwargs.

## What Changed

Rewired these scripts to call
`star_uvt_colorizers.build_feature_colorizer(cfg["colorize"], ...)`:

- `research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py`
- `research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py`
- `research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py`
- `research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py`
- `research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py`

This removes the repeated config kwargs for `hidden_dim`, activation,
pre-normalization, weight init, and init gain. The remaining direct
`FeatureToColor(...)` uses are lower-level prototype defaults, explicit tests,
or non-config factory paths.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  research_experiments/star_uvt_feature_tubes/dense_alpha_failure_diagnostic.py \
  research_experiments/star_uvt_feature_tubes/sparse_visual_loss_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py \
  research_experiments/star_uvt_feature_tubes/firstclass_backward_breakdown.py \
  research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py \
  src/train/star_uvt_colorizers.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_colorizers.py tests/test_star_uvt_checkpoints.py -q
```

Passed: `4 passed in 0.85s`.

The follow-up search over the touched diagnostics no longer finds
`from colorize import FeatureToColor`, `def _make_colorizer`, or
`FeatureToColor(`.

## Remaining Cleanup

This is a narrow dependency cleanup. The broader modularization goal remains
open; future slices should keep targeting duplicated contracts that are shared
by real trainers or active profiling scripts and can be covered by focused
runtime smokes.
