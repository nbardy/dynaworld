# Alpha Evolve Paper Reading Start

User goal: read 20 papers on the AlphaEvolve-style agentic code evolution topic
and take detailed notes, with 3-4 hours of attention per paper.

What changed:

- Started a durable paper notebook under `alpha_evolve/papers/`.
- Added `paper_queue.md` with 20 selected papers spanning AlphaEvolve/FunSearch,
  evolutionary algorithm design, reward-writing agents, general agentic search,
  SWE agents, and code-generation evaluation.
- Added `notes_template.md` so future notes keep the same evaluator-first shape:
  mechanism, evaluation, failure modes, DynaWorld mapping, and falsification
  tests.
- Added `synthesis.md` with the current working thesis: DynaWorld should evolve
  small microlibs behind hard evaluators, not broad trainer rewrites.
- Wrote the first detailed paper note:
  `alpha_evolve/papers/notes/001_alphaevolve.md`.

Current read state:

- Paper 001, AlphaEvolve, has a first detailed pass.
- Papers 002-020 are queued but not yet read deeply.
- This is not complete relative to the user's stated 20-paper, 3-4h-per-paper
  goal.

Important local conclusion:

- The local implementation should use candidate worktrees plus `codex exec`
  prompts, not assume `codex -p "..."` means prompt on this installed CLI. Here
  `-p` is profile selection.
- The first useful target should be a narrow microlib with deterministic gates
  and a metric cascade. Good early candidates are STAR UVT feature RGB-gradient
  handoff, mixed same-view/novel-view scheduling, Gaussian 512px guard heuristics,
  or evaluator/report generators.

Next read:

1. FunSearch, because AlphaEvolve directly extends it and it clarifies the
   single-function evaluator-grounded evolution baseline.
2. CodeEvolve, because it is the closest open AlphaEvolve-style implementation
   and should inform local runner/database shape.
3. Evolution of Heuristics or LLaMEA, because both expose prompt population,
   island, and archive design choices.
