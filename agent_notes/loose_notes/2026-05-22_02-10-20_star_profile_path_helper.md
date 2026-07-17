# STAR Profile Optional Path Helper Cleanup

## Context

The shared config utility already exposes `path_or_none(...)`, and most trainer
and STAR UVT paths use it for optional checkpoint/output paths. Two STAR UVT
profiling scripts still carried local `_path_or_none(...)` helpers with the
same behavior:

- `star_uvt_logit_handoff_rgb_vjp_profile.py`
- `star_uvt_feature1_wholegraph_profile.py`

## Changes

- Imported `path_or_none as _path_or_none` from `config_utils` in both scripts.
- Removed the local `_path_or_none(...)` definitions.
- Left profile logic, checkpoint loading, and report schemas unchanged.

## Validation

- `rtk uv run python -m py_compile research_experiments/star_uvt_feature_tubes/star_uvt_logit_handoff_rgb_vjp_profile.py research_experiments/star_uvt_feature_tubes/star_uvt_feature1_wholegraph_profile.py src/train/config_utils.py tests/test_config_and_dataset_io.py`
  passed.
- `rtk env PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_config_and_dataset_io.py -q`
  passed: `8 passed`.

## Notes

This is a config-boundary cleanup only. It does not change any profiling math or
the strict MPS/runtime requirements in those scripts.
