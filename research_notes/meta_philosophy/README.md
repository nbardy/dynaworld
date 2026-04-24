# Meta Philosophy

Strategic layer for the project. How to design architectures, how to avoid
slop, and what this project is actually trying to do.

## Reading order for a new agent

1. **`our_problem_core_requirements_and_goals_and_current_philosophy_and_insight.md`** — what the project is. Goals, data contract, inference requirements, unavoidable structure, failure modes F1–F7, ideas surfaced (as prior art, not targets). Load-bearing constraints.
2. **`architecture_design_north_star.md`** — the principle to design by. One-liner + six principles. Covers the mechanism families (objective / augmentation / architectural-seam / post-training / emergent) and why pre-committing to factorization is wrong.
3. **`world_splat_tokens_vs_observed_modality_tokens.md`** — why world/splat tokens are latent predictive assets, not text/image/video tokenizer targets. Read before proposing a tokenizer, splat-token target, or two-stage token prediction scheme.
4. **`how_to_think_about_architecture.md`** — mistakes to not repeat. 15+ named slop patterns with principles. Checklist before proposing. F1–F7 lookup. This is the "avoid regression to familiar-shaped bad answers" doc.
5. **`dynaworld_architecture_solution_prompt.md`** — DynaWorld-specific external-model prompt for generating 3-4 concrete architecture solutions around world/splat tokens, held-out source-video training, and non-degenerate splats.
6. **`how_prompt_guidance_could_have_been_better_for_model_architecture_research.md`** — how to ask external models for adjudication instead of mechanism-stacking. Use when designing the next research prompt.
7. **`chatgpt_pro_prompt_for_expert_divergent_web_of_thought_model_architecture_development.md`** — general architecture prompt for driving external strong-reasoning models. Required XML output format, methodology contract, forbidden moves. Paste the System Brief into the external model; attach the companion docs.

## Companion material elsewhere

- **`../framing_the_problem/`** — alternate framings of the core problem. Framing 1 is information-theoretic; framing 2 is contract-based; framing 3 is the patched predictive-quotient default. Read after the problem doc.
- **`../training_contract_v1.md`** — operational version of patched framing 3 for implementation and diagnostics.
- **`../three_architectures_for_novel_view_synthesis.md`** — concrete architecture design doc (A/B/C with diagrams, debate, pioneer pick). Read after the framings.
- **`../potential_directions_index.md`** — routing map for all research threads in this project.
- **`../../agent_notes/key_learnings.md`** — dense bank of surprising technical lessons (not strategic; tactical).

## Maintenance

- The docs here are the strategic core. They should stay small enough to read in one sitting.
- New strategic insights go here. Tactical technical learnings go to `agent_notes/key_learnings.md`. Session logs go to `agent_notes/loose_notes/`.
- `how_to_think_about_architecture.md` is append-only for new mistake classes. Do not delete entries.
- If a doc here grows beyond skimmable, compress, don't append.
