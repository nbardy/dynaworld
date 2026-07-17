# Tree Of Thoughts Paper Note

Context:
    Continued the `alpha_evolve/papers` reading queue with paper 011,
    Tree of Thoughts: Deliberate Problem Solving with Large Language Models.

Sources inspected:
    arXiv page and PDF for 2305.10601 and the official
    `princeton-nlp/tree-of-thought-llm` repo, including the BFS runner and task
    implementations for Game of 24, Creative Writing, and Mini Crosswords.

What changed:
    Added `alpha_evolve/papers/notes/011_tree_of_thoughts.md`, marked paper 011
    done in `paper_queue.md`, updated the README status, and extended
    `synthesis.md`.

Main lesson:
    ToT is a modular search interface: state, expand, heuristic evaluate,
    select, and backtrack. For `alpha_evolve`, thoughts should be auditable
    candidate artifacts such as patch plans, generated tests, prompt sections,
    and microlib drafts, not private reasoning text.

Design implications:
    Add microlibs for `thought_state_schema`, `candidate_expander`,
    `state_heuristic_evaluator`, `frontier_selector`,
    `backtracking_controller`, and `prune_archive`. Store pruned states instead
    of deleting them because ToT's crossword pruning result shows heuristic
    evaluators can be wrong.

Next queue item:
    Paper 012, Language Agent Tree Search. Read it as the next step from static
    ToT search into action-environment trajectories and MCTS-style agent search.
