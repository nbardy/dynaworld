# Self-Refine Paper Note

Context:
    Continued the `alpha_evolve/papers` reading queue with paper 010,
    Self-Refine: Iterative Refinement with Self-Feedback.

Sources inspected:
    arXiv page and PDF for 2303.17651, the NeurIPS-era official
    `madaan/self-refine` repo README, and implementation files for PIE code
    optimization and code readability.

What changed:
    Added `alpha_evolve/papers/notes/010_self_refine.md`, marked paper 010 done
    in `paper_queue.md`, updated the README status, and extended
    `synthesis.md`.

Main lesson:
    Self-Refine is useful as a bounded generate-feedback-refine helper. It is
    not enough as a correctness loop for repo code because same-model feedback
    often fails by pointing at the wrong location or suggesting the wrong fix.

Design implications:
    Add microlibs for `self_refine_loop`, `feedback_actionability_scorer`,
    `refinement_history_selector`, `soft_quality_refiner`, and
    `oracle_feedback_adapter`. Store every refinement iteration and select by
    evaluator score, not by "last draft wins."

Next queue item:
    Paper 011, Tree of Thoughts. Read it as explicit search over intermediate
    states and compare it to independent sampling plus Self-Refine retries.
