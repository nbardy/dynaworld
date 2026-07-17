# Data Loader Contract Cleanup

## What changed

- Added `research_notes/data_contract.md` as the canonical contract for the two
  current loader families:
  - single-sequence manifests in `src/train/sequence_data.py` for broad
    same-view scale pretraining;
  - multicam bundles in `src/train/multicam_video_data.py` for calibrated
    train-camera input and heldout-camera supervision.
- Linked the new contract from `research_notes/README.md` and the top-level
  `README.md`.
- Marked the first-pass single-camera and multicam data-collection items done
  in the README, and added the real remaining bridge: a mixed trainer/sampler
  that alternates same-view and novel-view losses.
- Added short pointers in the 1k dataset builder/config and the 1k train config
  so future agents do not mistake the scale pretrain manifest for the novel-view
  benchmark.

## Current data facts recorded

- Current broad train artifact:
  `data/single_video_pretrain/dynaworld_single_video_pretrain_1k_v0/train_manifest.jsonl`.
- Train rows are 1000 total:
  - 955 `single_view_video_window`
  - 44 `frame_clip_sequence`
  - 1 `synthetic_camera_json_sequence`
- Largest train sources are YouTube curated raw windows and ViVo single-view
  windows at 260 each, followed by Neural3D coffee at 193 and DeepView at 153.
- The generated 1k `eval_manifest.jsonl` is empty today. Heldout validation
  should use the separate single-camera test manifests and multicam val
  manifests named in the contract until the builder emits a non-empty eval pool.

## Decision

Do not invent another manifest format for "single + multicam" yet. The next
implementation should be a mixed data scheduler/trainer that samples from the
existing single-sequence loader and the existing multicam bundle loader, keeps
`same_view_recon` and `heldout_view_recon` separate in logs, and shares the
same V-JEPA/static-dynamic-token/splat decoder path.

## Verification

- Compiled `src/dataset_scripts/build_single_video_pretrain_manifest.py` with:
  `PYTHONPATH=src/train .venv/bin/python -m py_compile src/dataset_scripts/build_single_video_pretrain_manifest.py`.
