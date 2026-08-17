# Next Run Handoff — What to run when quota returns

**Trigger:** `codex_report.py --days 5 --top 5` shows weekly <30% and no `ultra` burst.

## 1) Metal kernel check (15 min)

```bash
python -c "from paper_training_types import MetalKernelSpec; print(MetalKernelSpec)"
# expect 3 specs: dynamic_gs, worldfoam_g4_v2, world_tubes_star_uvt
python src/train/kinetic_core/kernel_registry.py --list
```

If missing, implement `src/train/kinetic_core/*` adapters (≤300 l each) — do not copy `world_foam_lane2`.

## 2) Pilot (the only gate that may run first)

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --matrix research_experiments/paper_runner_suite/world_tubes_full_public_matrix_v1.jsonc \
  --filter "scene=CoffeeMartini seed=17" --lanes dynamic_gs,worldfoam_g4_v2,world_tubes_star_uvt \
  --pilot_only --verify --wandb_project dynaworld-paper-a
```

Must emit `pilot_only=true`, wall/mem receipts, and pass `codex_report --audit`.

## 3) Full evidence (fresh processes)

```bash
PYTHONDONTWRITEBYTECODE=1 .venv/bin/python research_experiments/paper_runner_suite/run_unified_paper_matrix.py \
  --matrix world_tubes_full_public_matrix_v1.jsonc --full --verify --wandb_project dynaworld-paper-a
# 36 G4-v2 rows + 6 F-scaling pts (F=4,8,16,32,64,128) + per-frame STAR replay baseline
```

Each row: `artifacts/run_*.json` verified by `verify_worldfoam_*` + `ArtifactStream`.

## 4) Charts + project page + videos

```bash
python research_experiments/paper_runner_suite/generate_world_tubes_paper_artifacts.py --from artifacts/ --out research_notes/gauged_uvt_trace_atlas/paper/
# emits CSV, plots (PSNR vs wall time, F-scaling), LaTeX tables
python web/dynaworld_browser_trainer/generate_project_page.py --artifacts artifacts/ --out web/dynaworld_browser_trainer/
ffmpeg -framerate 30 -pattern_type glob -i 'artifacts/frames/*_step*.png' -c:v libx264 -pix_fmt yuv420p artifacts/training_timelapse_side_by_side.mp4
```

Page shows 3 lanes side-by-side with synchronized training video + speed/quality scatter.

## 5) Paper (two venue files, collapse later)

Edit the existing venue sources (do not create PAPER_NEXT.md):
* Paper A: `research_notes/gauged_uvt_trace_atlas/paper/WORLD_TUBES_ICLR_MAIN_DRAFT.md` → `WORLD_TUBES_PAPER.tex`
* Paper B: `research_notes/worldfoam_paper/WORLD_FOAM_ICLR_MAIN_DRAFT.md` → `WORLD_FOAM_PAPER.tex`
Long dossiers `WORLD_*_PAPER_DRAFT.md` stay as math source. Collapse to one `PAPER.md` only if both ledgers flip to accepted.

Convert: `pandoc WORLD_TUBES_ICLR_MAIN_DRAFT.md -o WORLD_TUBES_PAPER.pdf` (or `latexmk` from tex); same for WorldFoam.

## Wandb + SNR + wall-clock

Log per step: `psnr/ssim/lpips`, `snr_proxy = 10*log10(var_signal/var_residual)`, `wall_ms{train,heldout,compile,forward,backward}`, `peak_mem`. Dashboard: `x=wall time, y=PSNR, color=lane, size=F`.

## Quick audit before pushing

```bash
python codex_report/codex_report.py --audit --no-cache  # expect gross≤3×net, rate 10–120 out/s
git -C dynaworld diff --stat HEAD                         # expect <2000 net per lane
```

Quoted goal for Codex prompt when resuming:
> Continue GOAL.md with sol 5.6 high + sol medium/luna high subs, 200M cap, KineticTrainer+MetalKernelSpec. Pilot seed-17 first, verify, then full matrix, charts, project page, wandb.
