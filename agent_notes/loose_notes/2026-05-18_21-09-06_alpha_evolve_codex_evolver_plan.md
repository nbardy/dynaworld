# Alpha Evolve Codex Evolver Plan

Date: 2026-05-18

This session created `alpha_evolve/` as the planning surface for a local
AlphaEvolve-inspired loop that uses non-interactive Codex candidate worktrees.

Paper read:

- AlphaEvolve centers the loop on executable evaluators, code edits, a program
  database, prompt context from prior scored programs, multi-metric scoring,
  evaluation cascades, and MAP-Elites/island-style diversity.
- The useful translation here is not "make a generic agent swarm"; it is
  "package each DynaWorld bottleneck as a small microlib with a hard evaluator."

Repo interpretation:

- Best first target: STAR UVT F32 feature RGB-gradient handoff. It has a clear
  measured bottleneck and existing JSON/parity/timing surfaces.
- Best second target: mixed same-view plus heldout scheduler. It is central to
  the world-token contract but needs a strict 1-step/10-step smoke.
- Other viable targets: Gaussian 512px promotion guard, V-JEPA/F32 multicam
  benchmark validators, WorldFoam Gate4 owner/candidate record compression,
  code-organization behavior helpers.

Important CLI note:

- Current local `codex -p` means `--profile`, not prompt. The runner should call
  `codex exec --cd "$WORKTREE" ... "$PROMPT"` or provide a wrapper that
  translates a user-facing `-p/--prompt` flag to `codex exec`.

Files added:

- `alpha_evolve/README.md`
- `alpha_evolve/codex_evolver_design.md`
- `alpha_evolve/problem_targets.md`
- `alpha_evolve/prompt_contract.md`
- `alpha_evolve/microlibs/README.md`
- `alpha_evolve/microlibs/star_uvt_feature_rgb_handoff.md`
- `alpha_evolve/microlibs/mixed_same_view_novel_scheduler.md`
- `alpha_evolve/microlibs/gaussian_512_promotion_guard.md`
- `alpha_evolve/microlibs/vjepa_multicam_benchmark_contract.md`
- `alpha_evolve/microlibs/worldfoam_gate4_records.md`
- `alpha_evolve/microlibs/code_org_helpers.md`

Next implementation step:

Build a tiny runner for exactly one microlib:

1. Create a disposable worktree from a known base.
2. Render the STAR feature prompt.
3. Run `codex exec` once.
4. Save patch/final/events.
5. Run only Stage 0 and Stage 1.
6. Insert one row into `programs.jsonl`.

Only after that should the loop grow islands, Pareto selection, meta prompts,
or parallel candidate evaluation.
