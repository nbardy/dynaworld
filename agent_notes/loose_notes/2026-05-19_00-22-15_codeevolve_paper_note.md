# CodeEvolve Paper Note

User goal remains active: read 20 papers on AlphaEvolve-style agentic code
evolution and write detailed notes for each with DynaWorld synthesis.

What changed in this chunk:

- Read CodeEvolve arXiv v4, the framework repository, and the experiment
  artifact repository.
- Added `alpha_evolve/papers/notes/003_codeevolve.md`.
- Marked paper 003 as done in `alpha_evolve/papers/paper_queue.md`.
- Updated `alpha_evolve/papers/README.md` and `synthesis.md`.

Main lesson:

CodeEvolve is the implementation bridge from FunSearch to AlphaEvolve. Keep the
first local proof as narrow function evolution, but design the database so it can
also represent bounded patch evolution with parent IDs, prompt IDs, ancestors,
inspirations, operator names, islands, migration origin, archive cells, stage
pass depth, and failure logs.

Specific DynaWorld implications:

- Add `direct_edit` and later optional `structured_patch` modes; Codex can edit
  directly now, while SEARCH/REPLACE may help replay high-risk microlibs later.
- Use immutable prompt contracts before meta-prompting: allowed paths,
  forbidden edits, evaluator commands, data-contract language, and final schema
  must not mutate.
- Start the CodeEvolve-shaped runner small: 3 islands, cycle migration, 2
  inspirations, depth 3, exploration around 0.2, and per-microlib behavior
  descriptors.

Next paper:

Evolution of Heuristics should be read next because it is the heuristic-design
lineage that CodeEvolve explicitly builds on and uses in its benchmark suite.
