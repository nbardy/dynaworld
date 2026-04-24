# Architecture prompt output hygiene

Session on 2026-04-24. User pasted a ChatGPT Pro response from the DynaWorld
architecture solution prompt and asked whether the prompt was working.

## Observed output quality

- Conceptually useful: the model understood the main philosophy, produced the
  expected A/B/C/D option families, kept query camera at the rasterizer
  boundary, and recommended single-stage held-out prediction first.
- Operationally broken: the response was not valid XML. It included
  `:contentReference[...]` citations, Markdown code fences inside XML, malformed
  tags, and control-character/terminal-escape garbage in a few places.
- The issue was prompt/output hygiene, not the core architecture context.

## What changed

- Tightened `research_notes/meta_philosophy/dynaworld_architecture_solution_prompt.md`
  with strict output hygiene:
  - valid parseable XML only,
  - no Markdown fences,
  - no citations or `contentReference` markers,
  - no control characters,
  - use CDATA for pseudocode,
  - check tag balance before finalizing.
- Updated the follow-up repair prompt to explicitly remove malformed artifacts.
- Copied a revised run-ready prompt to the clipboard with an explicit
  `EXECUTE NOW` question and valid-XML-only instruction.

## Verification

- Ran `git diff --check`; no whitespace errors.
