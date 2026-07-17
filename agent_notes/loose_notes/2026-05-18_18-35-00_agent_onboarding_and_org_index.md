# Agent Onboarding And Organization Index

## What changed

- Added `PROJECT_INDEX.md` as the new first operational index after `AGENTS.md`.
- Added `EXPERIMENTS.md` as the active experiment registry with configs, logs,
  result JSONs, W&B ids, and next actions.
- Added `CODE_ORGANIZATION.md` as the modularity/deduplication roadmap.
- Wired `AGENTS.md` and `TODO/README.md` to those docs.

## Why

A new agent had to infer project state from a mix of `AGENTS.md`, `TODO/README`,
`BASELINES.md`, loose notes, and scattered result files. That was too dependent
on thread memory. The new structure makes the read order explicit:

1. `AGENTS.md`
2. `PROJECT_INDEX.md`
3. `TODO/README.md`
4. `EXPERIMENTS.md`
5. `BASELINES.md`
6. `research_notes/data_contract.md`
7. `CODE_ORGANIZATION.md`
8. `agent_notes/key_learnings.md`
9. Latest relevant loose note

## Current organization decision

Keep `AGENTS.md` short as the rules/startup router. Put project map details in
`PROJECT_INDEX.md`, experiment lane details in `EXPERIMENTS.md`, and refactor
policy in `CODE_ORGANIZATION.md`.
