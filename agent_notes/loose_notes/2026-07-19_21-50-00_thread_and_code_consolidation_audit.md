# Thread And Code Consolidation Audit

Date: 2026-07-19

## What Was Reviewed

- Recent DynaWorld Codex tasks for World Tubes, Gauged/STAR UVT, WorldFoam,
  paper protocol work, and the browser trainer.
- Canonical project routing in `PROJECT_INDEX.md`, `TODO/README.md`,
  `EXPERIMENTS.md`, `CODE_ORGANIZATION.md`, and the renderer lane taxonomy.
- Current dirty-tree ownership.
- Focused mechanical health scans for `src/train` and
  `web/dynaworld_browser_trainer`.

## Decisions

- Archived the completed orbit-chart, STAR-review, and WorldFoam-resume tasks.
- The older handoff-review task also appears complete, but the first archive
  operation returned `Inactive thread archive did not persist`; leave that
  exact state visible rather than claiming it was archived.
- Kept the unread paper-closeout task visible.
- Kept the active browser task running and sent it a consolidation boundary:
  reuse canonical multicamera, paper-protocol, and browser-export contracts;
  do not create a parallel research/trainer hierarchy.
- Preserved all active paper-protocol and browser edits on disk.
- Wrote `TODO/2026-07-19_dynaworld_consolidation_program.md` as the bounded
  execution plan. It treats package boundaries, large owner modules, frozen
  research code, and the browser demo as separate cleanup clusters.

## Scan Caveats

The initial root scan was stopped because it entered a bundled Blender Python
distribution under `data/`; its findings are invalid. The focused scans are
useful, but their strict scores are not yet meaningful because subjective
dimensions were not reviewed. The browser scan also lacked ESLint coverage.

## Handoff

Do not start mass moves while the unified paper runner and browser multicamera
conversion are active. Once stable, begin with the paper protocol package
boundary and verify actual runtime paths before proceeding to trainer hotspot
splits.
