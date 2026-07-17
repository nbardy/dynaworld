# Codex HumanEval Paper Note

Created `alpha_evolve/papers/notes/017_codex_humaneval.md` as the seventeenth
paper note in the AlphaEvolve/agentic-code-evolution survey.

Sources inspected:

- arXiv paper `2107.03374`
- official `openai/human-eval` README
- official pass@k/evaluation implementation

Main takeaway for local `alpha_evolve` work:

- Functional correctness and unbiased pass@k should be the metric layer before
  any evolution claims.
- Sampled patch pools need both selected pass@1 and oracle pass@k/ranker-gap
  reporting.
- Generated code and generated tests are untrusted; the runner needs sandbox
  policy metadata and timeouts.

Queue state after this note:

- Papers 001-017 have first-pass detailed notes.
- Papers 018-020 remain queued, with the Program Synthesis paper next.
