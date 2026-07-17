# Reflexion Paper Note

Context:
    Continued the `alpha_evolve/papers` reading queue with paper 009,
    Reflexion: Language Agents with Verbal Reinforcement Learning.

Sources inspected:
    arXiv page and PDF for 2303.11366, plus the official
    `noahshinn/reflexion` repo. Relevant repo files included the ALFWorld
    reflection generator, HotPotQA agents/ReAct code, and programming loop files
    for Reflexion, Reflexion-UCS, Python generation, and Python execution.

What changed:
    Added `alpha_evolve/papers/notes/009_reflexion.md`, marked paper 009 done in
    `paper_queue.md`, updated the README status, and extended
    `synthesis.md`.

Main lesson:
    Reflection is useful as bounded, candidate-visible failure memory. It should
    never be the evaluator. The local alpha-evolve design should separate
    reflection, archive, evaluator, and verified skill library.

Design implications:
    Add microlibs for `verbal_reflection_memory`, `reflection_builder`,
    `reflection_invalidator`, `false_positive_guard`, and
    `reflection_budget_controller`. Generated tests can shape repair attempts,
    but promotion must stay with repo-owned gates because false-positive visible
    tests are the dangerous failure mode.

Next queue item:
    Paper 010, Self-Refine. Read it as the contrast case: iterative feedback and
    refinement without external execution as the primary feedback loop.
