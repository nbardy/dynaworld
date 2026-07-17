# Train CLI Path Argument Helper

## Goal

Continue the trainer-entrypoint cleanup by removing the last manual
one-config-argument check from the `src/train` CLI surface without changing how
the registry CLI dispatches configs.

## Change

- Added `train_cli.run_path_arg(...)` for entrypoints that need a raw config
  path instead of a preloaded config dict.
- Routed `src/train/train.py` through `run_path_arg(...)`.
- Kept `src/train/train.py` dispatching by path through
  `trainer_registry.run_config(...)`; this preserves the registry behavior and
  external-launcher error paths.
- Added focused tests for raw path forwarding and wrong-arity usage errors.

## Validation

Commands run from the Dynaworld root:

```bash
rtk .venv/bin/python -m py_compile src/train/train_cli.py src/train/train.py tests/test_train_cli.py
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_train_cli.py -q
```

The direct venv pytest command failed because pytest is not installed in the
project venv; the repo-standard `uv run --with pytest` command passed:

```text
6 passed in 0.01s
```

An `rg` check over `src/train` found no remaining manual `len(sys.argv)` or
`sys.argv[1]` config-entrypoint boilerplate outside the shared helper.

## Notes

This is intentionally a tiny entrypoint boundary cleanup. It does not change
trainer construction, config normalization, arch routing, or training behavior.
