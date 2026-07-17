# Microlib: Mixed Same-View Plus Heldout Novel-View Scheduler

## Problem

Build the smallest scheduler/trainer bridge that alternates broad same-view
single-sequence data with multicam heldout-camera supervision while logging
the losses separately.

## Why Now

This is the named bridge in `research_notes/data_contract.md`,
`PROJECT_INDEX.md`, `TODO/README.md`, and `CODE_ORGANIZATION.md`. The loaders
exist; the missing behavior is scheduler/trainer integration.

## Allowed Edits

Likely surface:

- `src/train/sequence_data.py`
- `src/train/multicam_video_data.py`
- `src/train/train_video_token_implicit_dynamic.py`
- `src/train/runtime_types.py`
- `src/train_configs/*.jsonc` for one smoke config
- `tests/` focused scheduler/loader tests

Avoid touching STAR UVT, PowerFoam, or WorldFoam code for this microlib.

## Baseline

Current state:

- same-view loader: `load_manifest_sequence` / `load_manifest_sequences`
- multicam loader: `load_multicam_video_bundle`
- no mixed trainer yet
- same-view and heldout metrics must not collapse into one unnamed number

## Evaluator Cascade

Stage 0:

- config validates
- no new manifest format
- no broad base trainer abstraction

Stage 1:

- tests prove both loader families can be sampled by one scheduler
- tests prove `loss_kind` or equivalent explicit label is present
- tests prove target/input exact overlap is rejected or impossible by
  construction

Stage 2:

- 1-step offline smoke exercises one same-view and one heldout batch
- output/log payload contains `same_view_recon` and `heldout_view_recon`
- both losses are finite

Stage 3:

- 10-step smoke with W&B enabled if the run will be compared
- media names make train/heldout explicit

## Primary Metrics

- both batch kinds executed
- finite losses
- separate log keys
- no target/input overlap
- minimal LOC/change surface

## Hard Rejects

- A third vague manifest format.
- Logging only `loss` with no separate `same_view_recon` and
  `heldout_view_recon`.
- Query camera entering a learned pre-render branch.
- Exact heldout target frames visible to encoder input.
- Large trainer inheritance framework.

## Promotion Gate

The first useful promotion is not a quality claim. It is a working mixed smoke
config and a loose note saying what it exercises. Benchmark claims still need
source/camera-disjoint manifests and `BASELINES.md` rows.
