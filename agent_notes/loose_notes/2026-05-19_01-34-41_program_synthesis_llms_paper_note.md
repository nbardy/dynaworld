# Program Synthesis With LLMs Paper Note

Created `alpha_evolve/papers/notes/018_program_synthesis_llms.md` as the
eighteenth paper note in the AlphaEvolve/agentic-code-evolution survey.

Sources inspected:

- arXiv paper `2108.07732`
- Google Research publication page
- official MBPP README and JSONL sample

Main takeaway for local `alpha_evolve` work:

- Build an MBPP-like local microlib task suite before full repo patch tasks.
- Track prompt-example seed, sample count, visible tests, hidden challenge
  tests, pass@k, sample reliability, and ranker gap.
- Do not trust visible tests alone; MBPP shows ordinary tests can over-credit
  narrow solutions that fail challenge tests.

Queue state after this note:

- Papers 001-018 have first-pass detailed notes.
- Papers 019-020 remain queued, with AlphaCode next.
