# GOAL — Next Quota Run

**Status:** queued until `codex_report --days 5` shows weekly <30% (quota resets ~08-23). Do not start before.
**Lead:** `sol 5.6 high` (not ultra) → subs `sol medium` (code/kernels), `luna high` (docs/charts)
**Token cap:** 200M — abort if `rate >120 out/s` or `gross>3×net`.

## One paper (three lanes, one trainer)

This GOAL does **not** re-define the paper. It executes the existing two-paper dossiers via one Metal runner.

*Canonical sources — read them, do not copy them:*
- Paper A plan: `TODO/world_tubes_paper_finish_master_plan_2026-08-13.md` — 9 runtime jobs (frozen F-sweep, variable-camera curve, 7 public contexts), gate `0/21 → 21`.
- Paper B plan: `TODO/worldfoam_memory_light_native4d.md` — `G6 0/21 / G4 0/36`, 133 schemas, pilot `F=8/64/300` then `F=8/64/300` matrix.
- Venue drafts: `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_ICLR_MAIN_DRAFT.md` and `research_notes/worldfoam_paper/WORLD_FOAM_ICLR_MAIN_DRAFT.md` (long dossiers `WORLD_*_PAPER_DRAFT.md` stay as math source).

*What this GOAL adds:* the **shared execution harness** that both papers now use (introduced after those plans):

- One `KineticTrainer(strategy: MetalKernelSpec)` under `src/train/kinetic_core/` (to be created, ≤300 l per strategy) — no new `world_foam_lane2/*.py`. Existing shared types: `paper_training_types.py:MetalKernelSpec`, `paper_training_protocol.py:PaperCostTracker/PaperPhaseTimer`.
- Three lane specs: `dynamic_gs` (baseline `train_splat_baseline.py` + FasterGS4D), `worldfoam_g4_v2` (1.2M selected pix, 196k heldout compiles), `world_tubes_star_uvt` (`F=4,8,16,32,64,128` vs per-frame STAR replay).

## What to actually run (points to the plans, not re-states them)

1. **Pilot `Coffee Martini seed-17` for all 3 lanes** — `run_unified_paper_matrix.py --matrix world_tubes_full_public_matrix_v1.jsonc --filter scene=CoffeeMartini,seed=17 --pilot_only --verify --wandb dynaworld-paper-a` → `pilot_only=true` + wall/mem receipts. This is the gate both Paper A `0/7` and Paper B `0/36` plans call “pilot”.

2. **Full matrix** — same command `--full --verify` over `research_notes/data_contract.md` scenes (Coffee triplets + 2 Neural3D + 1 D-NeRF). Paper A needs 7 rows min, Paper B needs 36. Run fresh MPS processes, verifier accepts `artifacts/*.json`.

3. **Charts + project page + video** — `generate_world_tubes_paper_artifacts.py --from artifacts/` (CSV/plots/LaTeX) + `web/dynaworld_browser_trainer/` side-by-side + `ffmpeg` timelapse (every N steps). Wandb `run=scene/seed/lane` with `PSNR/SNR vs wall`.

4. **Paper** — edit the *existing* venue sources above, not a new `PAPER_NEXT.md`. Start as two venue files; collapse to one `PAPER.md→WORLD_TUBES_PAPER.tex` only if both ledgers flip to accepted. One repro command + `BASELINES.md` rows.

## Gates (from the plans)

Gates are defined in those TODOs — this GOAL just orders them: pilot → full (21+36 rows) → `codex_report --audit gross≤3×net` → venue PDF. See `TODO/README.md:Current Project State` for `0/21`/`0/36` reset reason.

## Ownership (claim in EXPERIMENTS.md#active-lanes)

`sol medium`: `src/train/kinetic_core/**`, `run_unified_paper_matrix.py`; `luna high`: the two `*_ICLR_MAIN_DRAFT.md` + tex; `luna medium`: `artifacts/**`, wandb, `web/**`. No new `world_foam_lane2/*.py`, no `TODO/*.md>50` (use `worldfoam_status.json`), no parallel same-file patches, no `ultra` without `BUDGET.md`.
