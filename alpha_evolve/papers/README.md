# Agentic Code Evolution Paper Notebook

Date started: 2026-05-19

Scope: papers that can inform a Codex-driven AlphaEvolve-style loop for
DynaWorld. The emphasis is on executable feedback, program search, code repair,
agent memory, search over trajectories, and evaluation design.

The user's target depth is 3-4 hours per paper. This folder is structured for
that multi-day pass. Notes should not be shallow summaries; each note should
end in concrete implications for `alpha_evolve/`, especially microlib shape,
evaluator design, candidate databases, and failure modes.

## Files

- `paper_queue.md` - selected 20-paper queue, priorities, sources, status.
- `notes_template.md` - required structure for each paper note.
- `notes/` - one detailed note per paper.
- `synthesis.md` - cross-paper lessons, filled as notes accumulate.

## Reading Standard

For each paper:

1. Record bibliographic metadata and primary links.
2. Extract the mechanism, not just the headline result.
3. Identify the feedback signal and what it actually proves.
4. Map the paper to a DynaWorld `alpha_evolve` design decision.
5. Red-team the idea: metric hacking, leakage, cost, brittleness, and missing
   evaluator coverage.
6. Write falsification tests we can run locally.

## Current Status

- Paper 001, AlphaEvolve: first detailed pass written.
- Paper 002, FunSearch: first detailed pass written.
- Paper 003, CodeEvolve: first detailed pass written.
- Paper 004, Evolution of Heuristics: first detailed pass written.
- Paper 005, LLaMEA: first detailed pass written.
- Paper 006, Eureka: first detailed pass written.
- Paper 007, Voyager: first detailed pass written.
- Paper 008, ReAct: first detailed pass written.
- Paper 009, Reflexion: first detailed pass written.
- Paper 010, Self-Refine: first detailed pass written.
- Paper 011, Tree of Thoughts: first detailed pass written.
- Paper 012, LATS: first detailed pass written.
- Paper 013, SWE-bench: first detailed pass written.
- Paper 014, SWE-agent: first detailed pass written.
- Paper 015, Agentless: first detailed pass written.
- Paper 016, OpenHands: first detailed pass written.
- Paper 017, Codex/HumanEval: first detailed pass written.
- Paper 018, Program Synthesis with LLMs: first detailed pass written.
- Paper 019, AlphaCode: first detailed pass written.
- Paper 020, CodeT: first detailed pass written.

This completes the first detailed notebook pass over the selected 20-paper
queue. The notes are intentionally implementation-facing: each paper maps back
to `alpha_evolve/` microlibs, evaluator boundaries, candidate storage, and
failure modes for a Codex-driven evolution runner.
