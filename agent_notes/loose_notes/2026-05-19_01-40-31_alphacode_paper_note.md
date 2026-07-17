# 2026-05-19 01:40:31 - AlphaCode paper note

Context:
    Continued the 20-paper alpha_evolve survey with paper 019:
    "Competition-Level Code Generation with AlphaCode."

What changed:
    Added `alpha_evolve/papers/notes/019_alphacode.md`.
    Updated `alpha_evolve/papers/README.md` and `paper_queue.md` to mark paper
    019 done.
    Added an "After Paper 019 - AlphaCode" synthesis section.

Key takeaways:
    AlphaCode is most useful here as a selection-kernel paper, not as a model
    scaling paper. It separates large candidate generation, visible example-test
    filtering, generated-probe behavioral clustering, and a bounded hidden-test
    submission budget.

    The DynaWorld transfer is to build the first `alpha_evolve` runner around
    `k` Codex candidates, public filters, behavior signatures, cluster-aware
    selection, and hidden gates. The important report is `n@k` plus oracle
    `pass@k`, which exposes the ranker gap.

Design implications:
    Start with competitive-programming-shaped microlibs such as JSONC config
    normalization, metric aggregation, result-table parsing, renderer capability
    selection, prompt context packing, and evaluator fingerprinting.

    Do not spend on persistent islands or expensive trainer/renderer evolution
    until the small sample/filter/cluster/submission harness proves selection
    beats random under hidden gates.
