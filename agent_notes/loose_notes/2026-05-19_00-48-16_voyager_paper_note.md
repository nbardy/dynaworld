# 2026-05-19 00:48:16 - Voyager paper note

Session chunk: paper 007 in the `alpha_evolve` 20-paper pass.

What changed:

- Added `alpha_evolve/papers/notes/007_voyager.md`.
- Marked paper 007 done in `alpha_evolve/papers/paper_queue.md`.
- Updated `alpha_evolve/papers/README.md` from six notes to seven notes.
- Added an "After Paper 007" synthesis section in
  `alpha_evolve/papers/synthesis.md`.

Sources inspected:

- arXiv abstract page and PDF for `2305.16291`.
- Official Voyager project site.
- Official `MineDojo/Voyager` GitHub README.
- Current implementation surfaces: `voyager/voyager.py`,
  `voyager/agents/skill.py`, `voyager/agents/action.py`,
  `voyager/agents/critic.py`, `voyager/agents/curriculum.py`, and the public
  skill-library checkpoint layout.

Main lesson:

Voyager adds the reusable executable-memory layer. Local `alpha_evolve` should
separate the candidate archive from a verified skill library. Every attempt and
failure belongs in the archive. Only hard-gated, reusable artifacts should be
retrieved as executable skills in future prompts.

Important design consequences:

- Add `verified_skill_library`, `skill_retriever`, `repair_attempt_loop`,
  `bounded_task_scheduler`, and `critic_reflection_adapter` microlibs to the
  design map.
- Use Voyager's four-attempt repair loop shape inside a single evolutionary
  generation before archiving a candidate.
- Use LLM/human critic output for reflection and diagnosis, not pass/fail
  promotion.
- Retrieve skills by embedding plus structured filters: microlib kind, schema
  version, allowed paths, evaluator family, data mode, and dependency version.
- Treat curriculum as a bounded scheduler over registered tasks until the
  evaluator stack is mature enough for open-ended task proposal.

Next paper:

Paper 008 is ReAct. Read it as the minimal reasoning/action baseline that
Voyager and later software-agent papers extend.
