# STAR UVT Colorizer Helper Cleanup

## Context

Continuation of the trainer modularization goal after the STAR UVT feature
target, sparse target-grid, and chunked render helpers were split out. The next
small duplicated seam was `FeatureToColor` construction: the feature STAR
overfit trainer, target-grid RGB probe, rendered-feature RGB probe, and RGB
probe checkpoint loader each manually repeated the same constructor kwargs.

## Changes

- Added `src/train/star_uvt_colorizers.py`.
  - `build_feature_colorizer(...)` owns STAR UVT `FeatureToColor` construction
    from the normalized `colorize` config.
  - `set_module_trainable(...)` owns the simple freeze/train toggle used by
    probe loading and rendered-probe trainability selection.
- Rewired:
  - `src/train/train_star_uvt_feature_rgb_probe.py`
  - `src/train/train_star_uvt_rendered_feature_rgb_probe.py`
  - `src/train/train_star_uvt_feature_overfit.py`
- Preserved the important `hidden_dim=null` behavior for single-layer probes;
  the helper converts only non-null hidden dims to `int`.
- Added `tests/test_star_uvt_colorizers.py` for helper behavior:
  - `hidden_dim=None` stays a single-layer colorizer and produces RGB output.
  - train/eval plus `requires_grad` toggling moves together.
- Updated `CODE_ORGANIZATION.md` to list `star_uvt_colorizers.py` as the shared
  STAR UVT colorizer construction boundary.

## Validation

```bash
rtk env PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/star_uvt_colorizers.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_star_uvt_colorizers.py
```

Passed.

```bash
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_colorizers.py \
  tests/test_star_uvt_feature_target_adapter.py \
  tests/test_star_uvt_feature_rgb_probe.py -q
```

Passed: `45 passed in 1.77s`.

```bash
rtk rg -n "FeatureToColor\(" src/train/train_star_uvt*.py src/train/star_uvt_*.py
```

Only remaining constructor is in `src/train/star_uvt_colorizers.py`.

```bash
rtk git diff --check
rtk rg -n "[ \t]+$" \
  src/train/star_uvt_colorizers.py \
  src/train/train_star_uvt_feature_overfit.py \
  src/train/train_star_uvt_feature_rgb_probe.py \
  src/train/train_star_uvt_rendered_feature_rgb_probe.py \
  tests/test_star_uvt_colorizers.py \
  CODE_ORGANIZATION.md
```

Both passed. The trailing-whitespace scan exited with no matches.

## Remaining Modularization Work

- The STAR UVT overfit script still owns model construction, checkpoint payload
  schema, visual-support diagnostics, and the long training policy. Those are
  larger than this helper slice and should be split only when there is a second
  real consumer or a clear trainer-interface boundary.
- A likely next bounded slice is checkpoint payload helpers for STAR UVT
  model/colorizer save/load metadata, because the rendered-feature probe and
  feature-overfit trainer both already depend on that checkpoint contract.
