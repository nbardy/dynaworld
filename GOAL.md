# GOAL — Next Quota Run: Three-Lane Paper with Metal Kernels

**Status:** queued (quota 0% until reset ~2026-08-23) — do not start until `codex_report` shows <30% weekly.
**Lead effort:** `sol 5.6 high`  (not ultra)
**Sub-agents:** `sol medium` (code/kernels), `luna high` (docs/charts/project page), `luna medium` (wandb/video)
**Token cap:** 200M total (50M per lane) — abort if `rate >120 out/s` or `gross>3×net`.

## Objective (one sentence)

Ship **World Tubes (sub-linear raster)** vs **World Foam (retained-depth)** vs **Dynamic 3DGS baseline** — all via **Metal kernels** — trained, ablated, charted, wandb-logged, side-by-side project page + training videos, and a full `PAPER.md` ready for pdf/ICLR.

## Three Metal lanes (one trainer, many kernels)

All lanes use `KineticTrainer(strategy: MetalKernelSpec)` under `src/train/kinetic_core/` — no new `world_foam_lane2/*.py` files.

1. **Baseline — Dynamic 3DGS** (`MetalKernelSpec: dynamic_gs`)
   - Existing `train_splat_baseline.py` + `FasterGS4D` staged path, wall-clock timed, `apply_paper_dataset_contract` same as others.
2. **World Foam — retained-depth / visibility ablation** (`worldfoam_g4_v2`)
   - `paper_kinetic_*` bounded `1,228,800` selected pixels, `196,608` heldout compiles, `PaperCostTracker` for mem/time.
3. **World Tubes — sub-linear rasterization cost** (`world_tubes_star_uvt`)
   - `projective STAR UVT` compiled, `F = 4,8,16,32,64,128` scaling suite, per-frame STAR replay as causal baseline. This is the paper's core claim.

Each lane emits **one** `MetalKernelSpec` + `compiled_artifact` + `checkpoint` + `timing` receipt.

## Evidence to produce (fail-closed)

* **Matrix:** `protocol × seed × scene × camera` via `run_unified_paper_matrix.py` + `world_tubes_full_public_matrix_v1.jsonc`
  - Scenes: Coffee Martini progressive 512 seeds 17/29/43 + pixel-matched fixed 512 + global shuffle (seed 17 pilot first), + 2 Neural3D scenes + 1 D-NeRF
  - Per row: 300 optimizer steps @ 4 samples/step, 1024 pixels/sample (=1,228,800/3,686,400 scalars), heldout `384x512` spatial-major
* **Metrics per row (wandb):** PSNR/SSIM/LPIPS, SNR proxy, wall-clock train/heldout, peak device mem, compile/forward/backward ms, `PaperCostTracker` bytes, trace/fallback counts. All logged to wandb project `dynaworld-paper-a` with `run=scene/seed/lane` grouping and `PaperPhaseTimer` wall bars.
* **Ablations & charts:** Lane vs lane (PSNR vs wall time), F-scaling (World Tubes 6 pts + per-frame replay), retained-depth (World Foam vs 3DGS). CSV + `matplotlib` plots + LaTeX tables via `generate_world_tubes_paper_artifacts.py`.
* **Project page:** `web/dynaworld_browser_trainer/` + `research_notes/.../paper/` — three models side-by-side, training-over-time videos (side-by-side renders every N steps), speed-vs-quality scatter, all derived from verified `artifacts/*.json`.
* **Paper:** Single `PAPER.md` (`research_notes/worldfoam_paper/PAPER_NEXT.md` → `WORLD_TUBES_PAPER.tex`) covering theory (gauged ray), implementation (STAR UVT), baselines, systems comparison, ablation tables, and one `python run_unified_paper_matrix.py --seed 17` repro command. No new gauge terminology.

## Gates (in order, do not skip)

1. Pilot `Coffee Martini seed-17` for all 3 lanes — verify bitwise parity on bounded track set, wall/mem receipts, `pilot_only=true`.
2. Full 36-row G4-v2 + 6-pt F-scaling — fresh MPS processes, independent verifier accepts `artifacts/*.json`.
3. Charts + project page + wandb dashboards pass `codex_report --audit gross≤3×net`.
4. `PAPER.md` with locked manifests, `BASELINES.md` rows appended.

## File ownership (claim in EXPERIMENTS.md#active-lanes before patch)

* `sol medium`: `src/train/kinetic_core/**`, `research_experiments/paper_runner_suite/run_unified_paper_matrix.py`
* `luna high`: `research_notes/worldfoam_paper/PAPER_NEXT.md`, `PAPER.md`, `WORLD_TUBES_PAPER.tex`
* `luna medium`: `artifacts/**`, wandb run configs, `web/dynaworld_browser_trainer/**`

## Blocked

* No new `world_foam_lane2/*.py` — add `KernelStrategy` instead.
* No `TODO/*.md >50` lines — use `TODO/worldfoam_status.json` + `agent_notes/loose_notes/` append.
* No parallel same-file patches; `ultra` forbidden without `BUDGET.md`.

## Success snippet for prompt

> Use sol 5.6 high + sol medium/luna high subs, KineticTrainer+MetalKernelSpec, 200M cap, commit every 500 net lines with `codex_report --audit` and wandb link.
