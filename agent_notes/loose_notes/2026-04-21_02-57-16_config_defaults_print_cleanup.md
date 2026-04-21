# Config Defaults And Print Cleanup

## Context

The prebaked-camera trainer had long f-string status prints like:

```text
scale_init={model_cfg.get('scale_init', 0.05)}
```

That was a code smell for two reasons:

- defaults were duplicated at use sites instead of normalized once
- status print logic duplicated schema knowledge and could drift from model construction

## Changes

- Added AGENTS.md rules:
  - backward-compatible defaults belong in config load/normalization
  - runtime code should read normalized config with explicit keys
  - status prints should use dictionary/summary helpers rather than long hand-built f-string chains
- Added small config utility helpers:
  - `apply_defaults`
  - `select_keys`
  - `format_key_values`
- Updated `src/train/dynamicTokenGS.py`:
  - centralized model/render/train/logging defaults in one normalization boundary
  - normalized LR schedule and optimizer config once in `resolve_config`
  - removed config `.get(..., default)` use sites from runtime model construction, optimizer construction, render calls, and status prints
  - replaced the Gaussian head print with `print_key_values("Gaussian head", select_keys(...))`
  - added `build_model_from_config(...)` so config-to-constructor mapping is one small boundary

## Validation

- `py_compile` passed for `src/train/config_utils.py` and `src/train/dynamicTokenGS.py`.
- `rg` found no `model_cfg.get`, `train_cfg.get`, `render_cfg.get`, `logging_cfg.get`, `optimizer_cfg.get`, or `schedule_cfg.get` in `src/train/dynamicTokenGS.py`.
- Disabled-W&B 2-step smoke passed for the current 128px Taichi 8192-splat config.
- Disabled-W&B 2-step smoke passed for the legacy 32px dense config, confirming the compatibility defaults are being applied during config normalization.
- `git diff --check` passed for touched files.
