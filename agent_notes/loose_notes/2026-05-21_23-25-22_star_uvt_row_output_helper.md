# STAR UVT Row-Output Helper

## Context

The trainer-unification pass had already moved STAR UVT prediction media and
row JSON output into `src/train/star_uvt_outputs.py` for feature overfit and
RGB-probe paths. The RGB STAR video overfit script still kept a local
`json.dumps(row)` print/write block.

## Change

- Routed `src/train/train_star_uvt_video_overfit.py` through
  `write_row_json_and_print(...)`.
- Removed the local `json` import from the RGB STAR overfit script.
- Added coverage in `tests/test_star_uvt_outputs.py` that the helper writes
  and prints the same sorted pretty JSON payload.
- Updated `CODE_ORGANIZATION.md` and `TODO/trainer_landscape_unification.md`
  so the docs name RGB overfit as part of the shared STAR row-output contract.

## Validation

```bash
PYTHONPATH=src/train .venv/bin/python -m py_compile \
  src/train/train_star_uvt_video_overfit.py \
  src/train/star_uvt_outputs.py \
  tests/test_star_uvt_outputs.py

PYTHONPATH=src/train uv run --with pytest python -m pytest \
  tests/test_star_uvt_outputs.py \
  tests/test_star_uvt_config_keys.py \
  tests/test_train_cli.py -q
```

Both passed before this note was written.

## Remaining

This closes one small helper duplication. It does not mean the broader trainer
modularization goal is done. Remaining cleanup should focus on live duplicate
imports/helpers found by `rg`, especially diagnostic scripts that still import
the base trainer just to reach config/runtime helpers, plus any large untracked
STAR config matrix that should be either promoted, archived, or pruned.
