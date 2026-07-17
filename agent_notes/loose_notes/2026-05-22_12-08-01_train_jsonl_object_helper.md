# Train JSONL Object Helper

## Goal

Continue the active modularization goal by moving repeated train-local JSONL
object-row decoding into the same small I/O helper that already owns plain JSON
file reads.

## Change

- Added `json_io.load_jsonl_objects(...)`.
- It owns:
  - opening JSONL files with UTF-8
  - skipping blank lines
  - decoding rows
  - reporting malformed JSON as `Invalid JSONL record in path:line`
  - enforcing that every row is a JSON object
- Routed callers:
  - `sequence_data.load_manifest_entries(...)`
  - `multicam_val_data.load_multicam_val_manifest(...)`
- Kept semantic ownership local:
  - same-view manifests still default missing `split` to `"train"`
  - multicam validation manifests still default missing `split` to `"val"`
  - each loader still owns its no-records error message
  - `sequence_data` still owns its manifest-to-sequence contract

## Validation

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src/train:. .venv/bin/python -m py_compile \
  src/train/json_io.py src/train/sequence_data.py src/train/multicam_val_data.py \
  tests/test_config_and_dataset_io.py

PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_config_and_dataset_io.py \
  tests/test_sequence_data_single_frame.py \
  tests/test_multicam_video_data.py -q

PYTHONPATH=src/train:. uv run --with pytest python -m pytest \
  tests/test_mixed_data_scheduler.py tests/test_mixed_same_heldout_trainer.py -q
```

Results:

- `py_compile` passed.
- Data-loader focused tests passed: `25 passed in 2.65s`.
- Mixed scheduler/trainer tests passed: `9 passed in 1.80s`.
- A direct smoke confirmed `load_jsonl_objects(...)`, same-view split filtering,
  and multicam validation split filtering share the row decoder while preserving
  their split defaults.

## Handoff

This is still helper-boundary cleanup, not a quality/convergence result. The
larger trainer modularization goal remains active until the unified paths have
real benchmark and W&B/media evidence.
