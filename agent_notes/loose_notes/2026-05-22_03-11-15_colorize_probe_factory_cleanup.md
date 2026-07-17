# Colorize Probe Factory Cleanup

## Context

The Token-GS trainer already builds its colorizer through
`model_factories.build_colorizer(...)`, which validates unknown config keys and
normalizes view-conditioning and detach-view flags. `probe_colorize_init.py`
still hand-built `FeatureToColor` from the config, so future colorize knobs
could silently drift between the trainer and the diagnostic probe.

## Changes

- Routed `probe_colorize_init.py` through `model_factories.build_colorizer(...)`.
- Removed the probe-local `FeatureToColor` and `normalize_view_condition`
  construction block.
- Kept the probe's model build, one-forward render, diagnostics, and stdout
  reporting local.

## Validation

- `PYTHONPATH=src/train:. uv run python -m py_compile` passed for
  `probe_colorize_init.py`, `model_factories.py`, and `colorize.py`.
- Import smoke for `probe_colorize_init` passed.
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest
  tests/test_config_factory_helpers.py -q` passed with 15 tests.

The usual parent `pyproject.toml` warning appeared during `uv run`; commands
still exited 0.

## State

This is a diagnostic-probe plumbing cleanup only. It does not run the colorize
probe against a real config or make any training-quality claim.
