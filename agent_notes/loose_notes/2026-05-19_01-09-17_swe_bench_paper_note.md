# SWE-bench Paper Note

Context:
    Continued the `alpha_evolve/papers` reading queue with paper 013,
    SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

Sources inspected:
    arXiv page and PDF for 2310.06770, current SWE-bench README and benchmark
    page, and harness implementation files for evaluation and grading.

What changed:
    Added `alpha_evolve/papers/notes/013_swe_bench.md`, marked paper 013 done
    in `paper_queue.md`, updated the README status, and extended
    `synthesis.md`.

Main lesson:
    Agentic code evolution needs a SWE-bench-like local harness before search
    comparisons mean anything. The key contract is fail-to-pass repair plus
    pass-to-pass maintenance in a replayable sandbox.

Design implications:
    Add microlibs for `repo_task_schema`, `patch_prediction_format`,
    `fail_pass_grader`, `context_retriever`, `evaluation_sandbox`, and
    `task_instance_builder`. Treat localization as its own stage and run
    oracle-context ablations before blaming mutation/search.

Next queue item:
    Paper 014, SWE-agent. Read it as the interface/agent design built on top of
    SWE-bench-style repo tasks.
