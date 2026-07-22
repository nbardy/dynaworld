# DynaWorld Consolidation Program

Date: 2026-07-19

Status: execution plan. This document does not replace the renderer taxonomy,
paper protocol, experiment registry, or baseline table. It defines the cleanup
order across those surfaces.

## Target State

DynaWorld has one active publication spine and one separate demo surface:

1. `paper_training_types.py` and `paper_training_protocol.py` own shared
   dataset, sampling, stage, and cost contracts.
2. The unified paper runner adapts those contracts to World Tubes/STAR UVT,
   WorldFoam/PowerFoam, and dynamic 3DGS without merging their kernels or model
   state.
3. World Tubes is the primary renderer paper; STAR UVT is its implementation
   family. WorldFoam is a parked second-paper challenger. Dynamic 3DGS is a
   baseline. These are not peer projects with independent infrastructure.
4. The browser trainer is a thin interactive consumer of canonical camera,
   split, and export contracts. It is not a fourth trainer architecture or a
   source of paper evidence.
5. Historical experiment code remains reproducible but frozen. New production
   behavior does not depend on old verifier, probe, or one-off runner modules.

## Current Ownership Boundary

Do not begin broad moves while the two active changes are unsettled:

- Paper protocol and unified paper runner: active, publication-critical.
- Browser multicamera conversion: active, demo-only; must reuse canonical data
  and calibration semantics rather than create a parallel stack.

All older Gauged UVT, STAR UVT planning, and WorldFoam resume tasks are
superseded by `research_notes/renderer_lane_taxonomy.md`, the unified paper
pipeline, and their dated loose notes.

## Evidence From The Cleanup Scan

The focused Python scan covered `src/train` only: 155 files and roughly 46K
lines. Its mechanical score was 84/100; the strict score is not meaningful yet
because the 20 subjective dimensions have not been reviewed. The useful
signals are:

- `src/train` is a flat namespace with 128 files.
- test-health mapping was 44.1%, much weaker than code quality (90.4%) and
  duplication (95.9%).
- the largest owner modules are
  `star_uvt_feature_overfit_trainer.py` (3,392 lines),
  `gs_models/dynamic_video_token_gs_implicit_camera.py` (2,938),
  `token_gs_trainer.py` (1,864), `powerfoam_metal_trainer.py` (1,751),
  `multicam_video_data.py` (1,639), and
  `dynamic_powerfoam_metal_trainer.py` (1,602).
- `research_experiments/` contains roughly 144K Python lines. Its largest
  WorldFoam owner-run module is 7,568 lines. This is primarily historical
  research surface, not a candidate for wholesale promotion into `src/train`.
- the browser trainer has only three JavaScript files but about 2,736 lines;
  `trainerWebGpu.js` alone has 1,648. Its mechanical scan was limited because
  ESLint was unavailable.
- a root-level Python scan was invalid because it entered the bundled Blender
  distribution under `data/`. Tooling and future repo-wide checks need explicit
  data/generated/vendor exclusions.

## Execution Order

### C0: Freeze Names And Thread Ownership

- Keep `research_notes/renderer_lane_taxonomy.md` authoritative.
- Do not add public aliases for Gauged UVT, World Tubes, STAR UVT, PRT,
  WorldFoam, Gate4, or PowerFoam.
- Archive completed planning tasks; retain only the paper pipeline, browser
  demo, and a bounded paper-closeout task.
- Do not delete experiment evidence merely because its task is archived.

Exit gate: every active task points to one canonical lane and names the files
it owns.

### C1: Finish The Shared Paper Contract

- Complete and verify the in-flight unified runner before moving its modules.
- Keep representation-specific adapters explicit.
- Add no base trainer hierarchy and no vague unified data loader.
- Run the configured 512-wide paper rows before cleanup that could invalidate
  timing or quality evidence.

Exit gate: focused protocol tests, runtime smokes, exact cost ledger, W&B
provenance, and baseline rows are green.

### C2: Establish Real Package Boundaries

After C1 is stable, reduce the flat `src/train` namespace by behavior domain:

- `data/`: sequence, multicamera, validation, and manifest contracts.
- `protocols/`: paper sampling, stages, budgets, and typed payloads.
- `trainers/`: trainer owners only; thin entrypoints remain outside.
- `renderers/`: existing renderer boundary, including backend adapters.
- `diagnostics/`: metrics, media, inspection, and artifact verification.

Move one dependency cluster at a time. Preserve compatibility imports only
when a checked-in config, test, or external script still needs them, and give
each shim a removal condition.

Exit gate per cluster: import graph smoke, relevant focused tests, F=3 runtime
smoke, and F=32/multicam smoke when signatures or payloads change.

### C3: Split Hotspots Along Existing Contracts

Prioritize extraction by stable behavioral boundary, not line count alone:

1. Split STAR feature overfit orchestration from support policy, evaluation,
   and artifact/report construction.
2. Split the dynamic implicit-camera model into token decoding, camera state,
   Gaussian state, and forward payload assembly.
3. Split multicamera data parsing/calibration from frame decoding/cache and
   batch selection.
4. Split PowerFoam trainer orchestration from resampling/state migration,
   evaluation, and W&B/artifact output.
5. Keep `fast_mac.py` as the renderer facade while moving variant loading,
   projection/raster calls, and autograd strategy selection behind focused
   private modules.

Do not merge World Tubes, WorldFoam, and dynamic 3DGS model state in the name
of deduplication. Share protocols and reporting; retain representation
semantics.

### C4: Freeze Research Offshoots

- Add a machine-readable registry for active, parked, superseded, and frozen
  experiment directories before moving or deleting anything.
- Frozen directories may receive verifier repairs, but not new architecture
  branches.
- New reusable behavior must enter through a small owned module in `src/train`
  with a runtime consumer and behavior-level test.
- Keep result JSONs and negative findings addressable from existing notes.

Exit gate: no production import depends on a frozen one-off experiment runner,
and every active experiment has a config, result path, owner, and stop rule.

### C5: Thin The Browser Demo

After the current multicamera task settles:

- use one browser-bundle exporter and the canonical train/heldout camera split;
- keep calibration conversion in one adapter with a parity fixture against the
  Python data contract;
- split `trainerWebGpu.js` into GPU device/pipeline setup, shared 3D state,
  training dispatch, rendering, and readback/metrics modules;
- keep browser-only approximation choices visibly labeled and out of paper
  baseline code.

Exit gate: the demo loads one exported bundle, trains on the declared cameras,
never samples the heldout camera, and reports parity-tested camera metadata.

## Stop Rules

- No new renderer name without a falsified existing contract.
- No new trainer file when a leaf helper or adapter is sufficient.
- No mass file move while publication-critical runs are in flight.
- No cleanup claim based on `py_compile`; exercise the actual runtime path.
- No archive/delete of result-bearing research code until imports, artifact
  references, and rerun commands are preserved.
- No score claim from the current strict desloppify score until subjective
  review is completed and scan coverage is fixed.

## Immediate Next Action

Do not begin C2 package moves yet. The paper is the governing objective.

1. Isolate and commit the verified unified paper runner, protocol contracts,
   manifest, configs, tests, and exact submodule states.
2. Complete the paper evidence schema: LPIPS, peak memory, serialized storage,
   normalized phase timing, and representation-specific trace/event/fallback
   statistics.
3. Add a fail-closed matrix aggregator for protocol, seed, scene, and camera
   split.
4. Run the 512-wide Coffee Martini matrix, then public scene/camera breadth.
5. Resume C2 only after accepted paper rows are stable enough that file moves
   cannot invalidate or delay them.

The detailed submission chain and code-retention boundary are in
`agent_notes/loose_notes/2026-07-19_21-55-34_paper_closure_and_code_retention.md`.
