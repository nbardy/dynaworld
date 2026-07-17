# 2026-05-19 00:43:07 - Eureka paper note

Session chunk: paper 006 in the `alpha_evolve` 20-paper pass.

What changed:

- Added `alpha_evolve/papers/notes/006_eureka.md`.
- Marked paper 006 done in `alpha_evolve/papers/paper_queue.md`.
- Updated `alpha_evolve/papers/README.md` from five notes to six notes.
- Added an "After Paper 006" synthesis section in
  `alpha_evolve/papers/synthesis.md`.

Sources inspected:

- arXiv abstract page and PDF for `2310.12931`.
- Official Eureka project site.
- Official `eureka-research/Eureka` GitHub README.
- Current implementation surfaces: `eureka/eureka.py`, Hydra config, and prompt
  fragments for policy feedback, code feedback, execution-error feedback, and
  reward signatures.

Main lesson:

Eureka contributes the key metric-hacking boundary for local AlphaEvolve work:
generated code may shape training or summarize diagnostics, but an immutable
external fitness gate must select candidates. Eureka evolves reward code,
trains a policy under that reward, then evaluates the resulting policy against a
separate task metric. That separation should be copied for DynaWorld shaped
losses, score compressors, curricula, generated tests, and diagnostics.

Important design consequences:

- Build `context_pruner`, `component_trace_logger`, and
  `reward_reflection_builder` microlibs before evolving full loss/reward code.
- Require generated shaping helpers to return named components, not just one
  scalar.
- Feed Codex compact reflection from component traces, execution errors, and
  hidden gate status.
- Keep heldout split definitions, leakage checks, final acceptance metrics, and
  promotion thresholds outside the editable surface.
- Track execute rate separately from best score so best-of-K sampling does not
  hide a fragile prompt contract.

Next paper:

Paper 007 is Voyager. Read it for skill-library persistence, self-verification,
and long-horizon code execution memory, then compare its memory model against
Eureka's Markovian last-best reflection.
