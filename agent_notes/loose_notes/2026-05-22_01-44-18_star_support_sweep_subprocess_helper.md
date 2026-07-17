# STAR Support Sweep Subprocess Helper Follow-Up

## Context

Follow-up to the STAR report subprocess helper cleanup. The first pass routed
three trainer report scripts through the shared logged subprocess wrapper, but
left `support_birth_split_sweep.py` local because it had extra launch behavior:

- trainer runs set `STAR_UVT_TILE_CAPACITY`
- trainer runs default `WANDB_MODE=offline` without overriding an existing env
- dense-support diagnostics launch a different script with a custom `--case`
  fanout

The shared helper now supports both defaults and overrides, so that exception no
longer needed a local `subprocess.run(...)` copy.

## Change

- Extended `run_logged_subprocess(...)` with `env_defaults`.
- Passed `env_defaults` through `run_star_uvt_feature_trainer_subprocess(...)`.
- Routed `support_birth_split_sweep.py` trainer runs through
  `run_star_uvt_feature_trainer_subprocess(...)`.
- Routed the dense-support diagnostic command through `run_logged_subprocess(...)`.
- Added a focused test for env defaults and overrides in
  `tests/test_star_uvt_report_artifacts.py`.

## Validation

- `uv run python -m py_compile research_experiments/star_uvt_feature_tubes/report_artifacts.py research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py tests/test_star_uvt_report_artifacts.py`
- `PYTHONPATH=src/train:. uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py -q`
  - `7 passed`
- `PYTHONPATH=src/train:. uv run python research_experiments/star_uvt_feature_tubes/support_birth_split_sweep.py --dry-run --target-sources uncovered_brightness --reallocate-tubes 32 --support-radii 32 --tile-capacities 128 --out-base /tmp/dynaworld_support_birth_split_smoke`
- `git diff --check`

The `uv` commands printed the existing parent-workspace warning about
`/Users/nicholasbardy/git/gsplats_browser/pyproject.toml` missing a `[project]`
table, but completed successfully.
