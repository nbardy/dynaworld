# Train CLI CSV Int Helper

## Context

Continuation of the modular trainer/code cleanup goal. A live scan showed the
main train entrypoints already route config/path CLI handling through
`train_cli.py`, but the two train-local colorize probes still parsed
comma-separated seed lists locally:

- `probe_colorize_init.py`
- `probe_colorize_matrix.py`

This is small, but it is the same train-probe CLI shape and should not fork as
more probes are added.

## Changes

- Added `train_cli.parse_csv_ints(...)`.
- Routed both colorize probes' `--seeds` handling through the shared helper.
- Added focused `tests/test_train_cli.py` coverage for trimming and empty
  entries.

## Validation

```bash
rtk .venv/bin/python -m py_compile \
  src/train/train_cli.py \
  src/train/probe_colorize_init.py \
  src/train/probe_colorize_matrix.py \
  tests/test_train_cli.py
rtk env PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_train_cli.py -q
rtk env PYTHONPATH=src/train .venv/bin/python src/train/probe_colorize_init.py --help
rtk env PYTHONPATH=src/train .venv/bin/python src/train/probe_colorize_matrix.py --help
```

Results: train CLI tests passed (`7 passed`), and both probe help paths loaded.
No probe forward pass or training run was executed; this was a train CLI helper
cleanup.
