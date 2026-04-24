# Architecture docs inventory

Session on 2026-04-24. User asked to inspect the repo docs around prior
architecture work, constraints, and model setup, then identify starting points.

## What I read

- `AGENTS.md` and parent `../RTK.md` startup guidance.
- Parent system/product docs: `../ARCHITECTURE.md`,
  `../docs/ARCHITECTURE_SUMMARY.md`, `../CORE_GOAL.md`, and due reminders.
- Strategic DynaWorld docs:
  - `research_notes/README.md`
  - `research_notes/meta_philosophy/README.md`
  - `research_notes/meta_philosophy/our_problem_core_requirements_and_goals_and_current_philosophy_and_insight.md`
  - `research_notes/meta_philosophy/architecture_design_north_star.md`
  - `research_notes/meta_philosophy/how_to_think_about_architecture.md`
  - `research_notes/framing_the_problem/README.md`
  - `research_notes/framing_the_problem/framing_1.md`
  - `research_notes/framing_the_problem/framing_2.md`
  - `research_notes/framing_the_problem/framing_3.md`
  - `research_notes/training_contract_v1.md`
  - `research_notes/three_architectures_for_novel_view_synthesis.md`
  - `research_notes/potential_directions_index.md`
- Supporting docs and chronology:
  - `research_notes/KEY_ARCHITECTURE_DECISIONS.md`
  - `research_notes/SESSION_Q_AND_A_SYNTHESIS.md`
  - `research_notes/proposed_architectures/**`
  - `research_notes/single_step/**`
  - `research_notes/video_diffusion_loss/**`
  - `research_notes/Frontier_dynaworld/**`
  - `agent_notes/key_learnings.md`
  - selected loose notes around novel-view training, token agreement, predictive
    quotient patch, and video-token implicit scaling.

## Current orientation

- The current strategic default is `framing_3.md` plus
  `training_contract_v1.md`: one encoder exports `W0`, query time enters
  `G(W0, tau)`, query camera enters only the fixed rasterizer, training uses
  omitted-observation prediction under `D_var`, and rate/minimality selects a
  representative.
- `three_architectures_for_novel_view_synthesis.md` is still useful as a
  candidate/mechanism comparison, but it predates or conflicts with the stricter
  framing-3 baseline in places. Treat its Architecture C recommendation as a
  historical candidate, not the current contract.
- `framing_1.md` is best for deriving what a loss bounds; `framing_2.md` is best
  for implementation audits around self-sufficiency/frame-local leaks;
  `framing_3.md` is the default for new proposals.
- The runnable/local engineering context is now strongly shaped by fast-mac v5+
  renderer work and the 65k-splat known-camera baseline; implicit camera is not
  falsified in small settings, but the video-token full-scene contract remains
  behind known-camera at 128px.
- No new surprising lesson emerged from this read-only inventory, so
  `agent_notes/key_learnings.md` was not changed.
