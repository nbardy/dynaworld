# Colorize Matrix Factory Follow-Up

## Context

After routing `probe_colorize_init.py` through
`model_factories.build_colorizer(...)`, `probe_colorize_matrix.py` still
instantiated `FeatureToColor` directly for each sweep cell. The matrix probe is
synthetic, but it still exercises the Token-GS colorizer settings that later
become config knobs.

## Changes

- Routed each matrix cell through `model_factories.build_colorizer(...)`.
- Kept the matrix-specific sweep table local: hidden dim, pre-norm, init type,
  and init gain remain explicit probe axes.
- Removed the probe-local `FeatureToColor` import/construction path.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `probe_colorize_matrix.py`, `model_factories.py`, and `colorize.py`.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest
  tests/test_config_factory_helpers.py -q` passed with 15 tests.
- `rg` found no remaining direct `FeatureToColor` import or constructor in
  `probe_colorize_init.py` or `probe_colorize_matrix.py`.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This only aligns probe construction and validation with the Token-GS trainer.
It does not run the matrix probe on a real config or make a colorizer-quality
claim.
