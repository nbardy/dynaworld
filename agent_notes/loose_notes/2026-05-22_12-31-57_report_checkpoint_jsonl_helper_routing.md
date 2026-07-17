# Report, Checkpoint, And JSONL Helper Routing

## Goal

Continue the trainer-code modularization pass by routing remaining live helper
forks through existing narrow boundaries, without changing run schemas,
artifact names, or experiment semantics.

## What changed

- Single-video pretrain shell launchers now import `json_io.load_jsonl_objects`
  in their embedded Python blocks when reading complete manifest JSONL object
  files. This covers audits, load checks, cache-status counts, prebake totals,
  full-cache guards, and status reporting in:
  - `src/train_scripts/train_single_video_pretrain_100_64f.sh`
  - `src/train_scripts/train_single_video_pretrain_300_64f.sh`
  - `src/train_scripts/train_single_video_pretrain_all_youtube_64f_512.sh`
- STAR feature-tube reports now have
  `report_artifacts.load_optional_report_json_or_error(...)` for tolerant
  report-object reads that preserve `{"_load_error": ...}` rows. The V-JEPA
  bridge audit uses it, and the dense-alpha failure diagnostic uses
  `checkpoint_utils.load_checkpoint_mapping(...)` instead of direct
  `torch.load(...)`.
- Dynamic Foam diagnostics and verifiers now route report-shaped JSON objects
  through `report_artifacts.load_report_json(...)` /
  `write_report_json(...)`, and route checkpoint payloads through
  `checkpoint_utils.load_checkpoint_mapping(...)` plus
  `model_state_dict_from_checkpoint(...)` where the script needs model weights.

## Boundaries kept local

- Dynamic Foam upstream-runner settings, copied remote JSON files, JSON row-list
  artifacts, PLY metadata, ffprobe output, and embedded remote smoke code stay
  local because those are not the shared Dynaworld report-object contract.
- STAR CSV streaming and plain log files stay local because they are not
  report-shaped JSON/text artifacts.
- The single-video launchers still own their action-specific stdout payloads and
  shell orchestration; only complete JSONL object decoding moved to the shared
  train helper.

## Validation

- `bash -n` passed for the three touched single-video pretrain launchers.
- `PYTHONPATH=src/train PYTHONDONTWRITEBYTECODE=1 .venv/bin/python -m py_compile`
  passed for the touched Dynamic Foam scripts and STAR report/diagnostic
  scripts.
- `PYTHONPATH=src/train uv run --with pytest python -m pytest tests/test_star_uvt_report_artifacts.py tests/test_checkpoint_utils.py -q`
  passed with `17 passed`.
- `git diff --check` passed for the touched paths.

## Remaining cleanup

The highest-value remaining `src/train` cleanup is the live trainer inheritance
chain where CLI-named trainer modules still import each other for base classes.
That is a larger, higher-risk module split and should be gated with registry
tests plus real F=3/F=32/multicam smokes, not slipped into a report-helper
cleanup pass.
