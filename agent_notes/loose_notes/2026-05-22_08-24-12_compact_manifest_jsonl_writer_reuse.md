# Compact Manifest JSONL Writer Reuse

## Context

The live duplicate scan still found a local `write_jsonl(...)` in
`src/dataset_scripts/build_single_video_pretrain_manifest.py`. The project
already has `train_artifacts.write_jsonl(...)`, but the local manifest writer
used compact separators to avoid extra bytes in large JSONL manifests.

## Changes

- `src/train/train_artifacts.py`
  - Added `compact: bool = False` to `write_jsonl(...)`.
  - Default behavior stays unchanged.
  - `compact=True` emits rows with `separators=(",", ":")`, preserving the
    single-video manifest builder's old row formatting.
- `src/dataset_scripts/build_single_video_pretrain_manifest.py`
  - Imports `write_jsonl` from `train_artifacts`.
  - Deletes its local writer.
  - Writes the full/train/eval/heldout manifests with
    `write_jsonl(..., compact=True)`.
- `tests/test_train_artifacts.py`
  - Added a compact JSONL assertion so the manifest-preserving behavior is
    protected at the shared helper boundary.

## Validation

- `rtk .venv/bin/python -m py_compile src/train/train_artifacts.py src/dataset_scripts/build_single_video_pretrain_manifest.py tests/test_train_artifacts.py`
- `rtk sh -lc 'PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_train_artifacts.py -q'`
  - `7 passed`
- `rtk .venv/bin/python src/dataset_scripts/build_single_video_pretrain_manifest.py --help`
- `rtk git diff --check -- src/train/train_artifacts.py src/dataset_scripts/build_single_video_pretrain_manifest.py tests/test_train_artifacts.py CODE_ORGANIZATION.md TODO/trainer_landscape_unification.md agent_notes/loose_notes/2026-05-22_08-24-12_compact_manifest_jsonl_writer_reuse.md`

## Notes

This removes a real duplicate without changing the generated manifest schema or
its compact row format. It is not a training-result change.
