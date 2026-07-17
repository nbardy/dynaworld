# Train Artifact Helper

## Context

The PowerFoam-family trainers repeated two file-output primitives:

- `output_dir.mkdir(...)` plus `resolved_config.json` writing with
  `serialize_config_value(cfg)`.
- Local `append_jsonl(...)` helpers in PowerFoam Metal and Dynamic PowerFoam
  Metal.

Checkpoint saving stays local because payloads and atomicity differ by trainer.

## Change

- Added `src/train/train_artifacts.py`.
- Added `write_resolved_config(output_dir, cfg)`.
- Added `append_jsonl(path, payload)`, using `serialize_config_value(...)` and
  stable key order.
- Updated PowerFoam Direct, Dynamic Gauge Foam, Dynamic PowerFoam Metal, and
  PowerFoam Metal to use the shared resolved-config writer.
- Updated Dynamic PowerFoam Metal and PowerFoam Metal to use the shared JSONL
  appender and removed their local appender functions.
- Added `tests/test_train_artifacts.py`.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_artifacts.py \
  src/train/train_powerfoam_direct.py \
  src/train/train_dynamic_gauge_foam.py \
  src/train/train_dynamic_powerfoam_metal.py \
  src/train/train_powerfoam_metal.py \
  tests/test_train_artifacts.py
```

Passed.

```bash
PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_train_artifacts.py tests/test_train_devices.py tests/test_train_logging.py -q
```

Result: `16 passed in 3.66s`.

## Remaining

This intentionally does not unify checkpoint saves. PowerFoam Direct uses
`atomic_torch_save`, Dynamic Gauge and Dynamic PowerFoam currently use
`torch.save`, and PowerFoam Metal has best/final checkpoint metadata. A helper
there would need to preserve real payload differences first.
