# OpenHands Paper Note

Created `alpha_evolve/papers/notes/016_openhands.md` as the sixteenth paper
note in the AlphaEvolve/agentic-code-evolution survey.

Sources inspected:

- arXiv paper `2407.16741` v3
- current OpenHands README

Main takeaway for local `alpha_evolve` work:

- Do not start by cloning a full generalist agent platform.
- Borrow OpenHands platform primitives only where they make the Codex evolver
  reproducible: typed event streams, runtime adapters, sandbox workspaces,
  evaluator registry, benchmark runner, cost tracker, and runner QC tests.
- Agentless remains the better first SWE-style baseline; OpenHands informs the
  infrastructure once the runner compares multiple tasks and variants.

Queue state after this note:

- Papers 001-016 have first-pass detailed notes.
- Papers 017-020 remain queued, with the Codex/HumanEval paper next.
