# SWE-agent Paper Note

Created `alpha_evolve/papers/notes/014_swe_agent.md` as the fourteenth paper
note in the AlphaEvolve/agentic-code-evolution survey.

Sources inspected:

- arXiv paper `2405.15793`
- official SWE-agent GitHub README
- SWE-agent configuration docs page

Main takeaway for the local `alpha_evolve` plan:

- SWE-agent is mostly an interface-design paper for this project. It says the
  command set, context packet, file-view shape, edit schema, guardrails, and
  history compression are part of the optimizer.
- For a Codex-driven evolver, the transfer is not to clone SWE-agent's exact
  commands. It is to wrap `codex exec "<prompt>"` with a crisp task packet,
  patch contract, evaluator report, patch guard, observation digest, and
  trajectory archive.
- The note adds candidate microlibs: `aci_contract`, `context_packet`,
  `search_summary`, `patch_guard`, `observation_digest`, `history_compactor`,
  `trajectory_store`, `budget_controller`, `failure_classifier`, and
  `interface_ablation_runner`.

Queue state after this note:

- Papers 001-014 have first-pass detailed notes.
- Papers 015-020 remain queued, with Agentless next.
