# 2026-05-19 00:33:52 - LLaMEA paper note

Session chunk: paper 005 in the `alpha_evolve` 20-paper pass.

What changed:

- Added `alpha_evolve/papers/notes/005_llamea.md`.
- Marked paper 005 done in `alpha_evolve/papers/paper_queue.md`.
- Updated `alpha_evolve/papers/README.md` from four notes to five notes.
- Added an "After Paper 005" synthesis section in
  `alpha_evolve/papers/synthesis.md`.

Sources inspected:

- arXiv abstract page and PDF for `2405.20132`.
- Official/current `XAI-liacs/LLaMEA` GitHub README.
- Current LLaMEA docs/project page.
- Selected current implementation surfaces: `llamea/llamea.py`,
  `llamea/solution.py`, and `examples/minimum_example.py` through raw GitHub.

Main lesson:

LLaMEA is the right first local runner shape, before islands or MAP-Elites. It
is a serial generate/evaluate/mutate loop with a hard evaluator, compact history
summary, selected parent code, structured score/error feedback, and either
best-so-far `(1+1)` or latest-parent `(1,1)` selection. For DynaWorld this maps
cleanly to a `llamea_serial` runner that wraps `codex exec`, candidate
worktrees, and one bounded microlib. The initial goal used the shorthand
`codex -p`, but the checked local CLI uses `-p` for `--profile`, so the adapter
should call the real non-interactive prompt surface.

Important design consequences:

- Beat independent `codex exec` sampling before claiming evolution works.
- Keep full logs out of the prompt; give the prompt a scalar score, compact
  score signature, and short error summary.
- Store errors as candidate data: stage failed, command, return code, timeout,
  exception summary, stderr tail, log path, and selection score.
- Use AOCC-style staged scores when early progress matters.
- Treat current LLaMEA features such as diff mode, HPO, niching, population
  evaluation, timeouts, and adaptive prompts as a roadmap after the serial loop
  proves signal.

Next paper:

Paper 006 is Eureka. Read it for evaluator/reward-code generation risk: how to
let a model write executable reward functions without letting it corrupt the
metric or overfit the simulator.
