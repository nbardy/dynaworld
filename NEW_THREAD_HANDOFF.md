# New Thread Handoff — Start Here

**Date:** 2026-08-17  **Branch:** `codex/repository-audit-cleanup` (parent `gsplats_browser` 180b61f → dynaworld 379c877)

## TL;DR for the new thread

Read in order: `GOAL.md` (thin pointer, 1 page) → `TODO/README.md` → `world_tubes_paper_finish_master_plan_2026-08-13.md` (Paper A, 9 jobs) + `worldfoam_memory_light_native4d.md` (Paper B, G6 0/21/G4 0/36) → `AUG_16TH_FOLLOWUP_SWARM_AUDIT.md` (what burned quota) → `NEXT_RUN_HANDOFF.md` (commands). Do not start until `codex_report --days 5` shows weekly <30%.

## What we did (quota burn + audit + tool)

* Overnight 08-15: 153 rollouts, 107 forked from `019f7a66` (`goal: work all night … ICLR`, `goal-objective.md: World Tubes …`), `sol 5.6 ultra` default, no cap. Gross `7455 patches / 163k loc_add` but net `59 files +20k / parent 44 +4k` — 3 monsters (`kinetic_dense_cached 6918l`, `TODO ledger 1630l`, `multicam_heldout 6991l`) each `~6k` gross but `0` net (rewritten identically by Sagan/Newton/Laplace). Naive `174B` cumulative → real stream dedup (`seenKeys` + `forkCutoff 5s`) `10.89B` (94% double-count).
* Built `codex-report/codex_report.py` (streaming dedup, `resolveModel`, `isValidCodexSession` head+tail 800, `patches/loc/toolSequence`, `activeMs` from `task_started/complete`, `_fix_rate` → 2–10 out/s not 5400) + `codex_scrape.py` (patch LOC per thread) with `--audit` gross vs net bar (`>3×` warn) and `XDG_CACHE_HOME||/tmp` cache. `py_compile` ok.
* Audited: `AUG_16TH_FOLLOWUP_SWARM_AUDIT.md` (259 l, refined with thread quotes `019f7a66` 26d, Laplace `01a002dd /root/g6_…`, token payload `1.715B/1.682B cached/4.2M out`, commits `026c130/3e698e8/8b9cb19→57a3062`, numstat tops `6918` etc.). Committed as `pre-dedup` baseline.
* Fixed review nits: cache path, `isValid 5→800` + `token_count` fallback, model filter `gpt-*` only, `locals()` leak in scrape, `head+tail` LOC undercount noted.
* Queued next run: `GOAL.md` now uses a 2M-token hard cap, one lead plus at
  most two non-recursive subagents, sequential accelerator work, pilot-first
  evidence gates, and fail-stop behavior. “ICLR-level” is explicitly treated
  as a quality bar rather than a selected venue.

## Existing 2-paper dossiers (do not rewrite, just execute)

* Paper A: `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_ICLR_MAIN_DRAFT.md` (28k, gated `frozen_world`) + `WORLD_TUBES_PAPER_DRAFT.md` (68k), `WORLD_TUBES_EXPERIMENT_PLAN.md` (26k), `REPRODUCE.md` (18k).
* Paper B: `research_notes/worldfoam_paper/WORLD_FOAM_ICLR_MAIN_DRAFT.md` (30k, gated `G0–G5`) + `WORLD_FOAM_PAPER_DRAFT.md` (62k), `WORLD_FOAM_MATH_APPENDIX.md` (21k), `WORLD_FOAM_MEMORY_LIGHT_THEOREM_LEDGER_2026-08-03.md` (120k).
* Master plans: `world_tubes_paper_finish_master_plan_2026-08-13.md` (68k, `Scope: Paper A only, 9 jobs`) and `worldfoam_memory_light_native4d.md` (105k). Zero rows accepted under schema-v2 (`0/7` Coffee Martini min, `0/21` Paper A, `G6 0/21 / G4 0/36`), hence next run is pilot-first.

## Next run — use when quota <30%

**Lead:** high reasoning; at most two medium-reasoning evidence operators; no
recursive spawning or ultra reasoning. **Hard cap:** 2M goal-accounted tokens,
with new work stopped at 1.6M.

1. Use the exact preflight and execution commands in the two canonical master
   plans; do not invent a shared-harness prerequisite before running evidence.
2. Paper A first: focused gate → evidence smoke → frozen sweep → bounded
   variable-camera curve → seven public contexts.
3. Paper B second: dry plans → guarded rebuild → G4 two-route pilot → G6/G4
   matrices only after their gates pass.
4. Regenerate tables and update the existing paper drafts only from accepted
   artifacts. Project-page and venue-template work are out of scope.

Ownership is artifact-based and disjoint: one operator for Paper A runtime
evidence, one for Paper B runtime evidence, and the lead alone for shared
status/manuscript files. No new audit, verifier, plan, renderer, or broad
refactor during the run.

## Commands to verify before pushing

```bash
python codex_report/codex_report.py --audit --no-cache   # gross≤3×net, 10–120 out/s
git -C dynaworld diff --stat HEAD                          # <2000 net per lane
python -m py_compile codex_report/codex_report.py codex_report/codex_scrape.py
```

## Commits to cite in new thread

`dynaworld: 026c130 audit baseline → 3e698e8 checkpoint (+20k) → e5f3ff3 refined (259 l) → 4c102de GOAL+handoff → 379c877 deduplicated GOAL/TODO` + parent bumps `8b9cb19 → 57a3062 → 180b61f`. Prior 185-l version at `026c130:AUG_16TH…`.

All info is now in `GOAL.md`, `NEXT_RUN_HANDOFF.md`, `TODO/README.md`, `AUG_16TH_FOLLOWUP_SWARM_AUDIT.md`, and this file — nothing else needed to restart.
