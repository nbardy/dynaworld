# Dataset Script Path Helper

## Goal

Remove the remaining small repo-root/train-path bootstrap copy from live dataset
CLIs without changing their data contracts.

## Change

- Added `src/dataset_scripts/script_paths.py` with:
  - `REPO_ROOT`
  - `ensure_train_path()`
  - `repo_path(...)`
  - `repo_text(...)`
- Updated `src/dataset_scripts/build_single_video_pretrain_manifest.py` to use
  the helper for train-path setup, repo-relative path resolution, and
  repo-relative display strings.
- Updated `src/dataset_scripts/visualize_multicam_rig.py` to use the helper
  for train-path setup and output path display strings.
- Routed complete JSON/text artifacts through `train_artifacts`:
  - manifest-builder dataset summary JSON uses `write_json(...)`
  - rig visualizer HTML uses `write_text(...)`
  - rig visualizer pose JSON uses `write_json(...)`
- Left the manifest JSONL writer local because it intentionally emits compact
  newline-delimited records with stable key ordering.

## Validation

- `rtk .venv/bin/python -m py_compile src/dataset_scripts/script_paths.py
  src/dataset_scripts/build_single_video_pretrain_manifest.py
  src/dataset_scripts/visualize_multicam_rig.py` passed.
- `rtk sh -lc 'PYTHONPATH=src/train .venv/bin/python
  src/dataset_scripts/build_single_video_pretrain_manifest.py --help'` passed.
- `rtk sh -lc 'PYTHONPATH=src/train .venv/bin/python
  src/dataset_scripts/visualize_multicam_rig.py --help'` passed.
- A focused `rg` scan now finds the dataset repo-root and train-path bootstrap
  only in `script_paths.py`.
- A focused artifact-write scan confirms complete JSON/text writes in the two
  dataset CLIs go through `train_artifacts`; the remaining direct writer is the
  compact manifest JSONL helper.

## Notes

This is intentionally smaller than a general project-wide path module.
Dataset-script path display and `src/train` import bootstrapping are one narrow
CLI concern; train-local third-party paths remain under `external_paths.py`, and
Dynamic Foam/Gauge experiment paths stay in their experiment-local helpers.
