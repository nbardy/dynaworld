# Project State And TODO Routing

## What changed

- Added a `Project State And TODO Routing` section to `AGENTS.md`.
- Added `TODO/README.md` as the active backlog index.

## Why

`AGENTS.md` already told agents where baselines and notes live, but it did not
name the `TODO/` folder or give a clear read order for "where are we and what is
next." That made future work too dependent on reading loose notes or guessing
from the latest dirty files.

## Current routing contract

For non-trivial work, future agents should start with:

1. `README.md` progress checklist.
2. `TODO/README.md` backlog map and active next steps.
3. `BASELINES.md` rerun TODOs before benchmark claims.
4. `research_notes/data_contract.md` before data/training changes.
5. `agent_notes/key_learnings.md` plus the latest relevant loose notes.

## Current next steps recorded

- Build the mixed same-view plus novel-view sampler/trainer.
- Add or document heldout/eval semantics for the 1k same-view lane.
- Promote the multicam V-JEPA/static-dynamic-token path into a benchmark
  contract with smoke gates and `BASELINES.md` rows.
- Benchmark direct-atomic dynamic splats across frame count separately from
  STAR-UVT.
