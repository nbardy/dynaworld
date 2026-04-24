# DynaWorld architecture solution prompt

Session on 2026-04-24. User asked whether we have a way to present a
"chain-of-thought prompt" to an external model that includes the project
philosophy, strong problem definition, and a method for generating 3-4
architecture solutions.

## What changed

- Added
  `research_notes/meta_philosophy/dynaworld_architecture_solution_prompt.md`.
- Linked it from:
  - `research_notes/meta_philosophy/README.md`
  - `research_notes/README.md`

## Prompt stance

- The prompt does not ask for unrestricted hidden chain-of-thought. It asks for
  compact, auditable rationale fields before each architecture branch:
  invariant targeted, supervision mechanism, anti-degeneracy argument, costs,
  falsification tests, and backtrack conditions.
- It includes a compressed DynaWorld context pack: deployment contract,
  `O/H/Q` training setup, token philosophy, source-video support assumptions,
  failure modes, diagnostics, and forbidden moves.
- It forces 3-4 structurally different architecture branches and synthesis to
  one recommendation rather than a menu.

## Verification

- Ran `git diff --check`; no whitespace errors.
- Counted the prompt at 542 lines and checked the heading outline with `rg`.

No code tests were run because this was documentation-only.
