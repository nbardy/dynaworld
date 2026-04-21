# Dynaworld Agent Guide

## Project Skills

Project-local Codex skills live under `.agents/skills/`. Use them when their
names or trigger phrases match the user request.

- `.agents/skills/deep-critical-thought/SKILL.md`: Use for durable thinking
  documents, "continue" expansion passes, branching/backtracking analysis,
  proofs, derivations, red-team hypothesis challenges, and rigorous agent notes.

## Agent Notes

There are two agent memory layers. Use both, but do not blur them.

### Loose Notes

Put raw session journals in `agent_notes/loose_notes/`.

These are append-only progress logs. They are not curated knowledge and they are
not only for what worked. They are the long-form lab notebook for the messy
process:

- what we did
- why we thought it made sense at the time
- ideas and assumptions that turned out wrong
- commands, experiments, and benchmarks that changed our mind
- bugs, surprises, failed attempts, and fixes
- stray technical thoughts that may matter later
- handoff context for future agents

Write one loose note per meaningful session or work chunk. Do not silently
rewrite old history; add a new note when the understanding changes. Small typo
fixes are fine, but the journal should preserve the trail.

Loose note filenames must use searchable datetime-first slugs:

```text
agent_notes/loose_notes/{YYYY-MM-DD_HH-MM-SS}_{topic_slug}.md
```

Examples:

```text
agent_notes/loose_notes/2026-04-20_12-57-46_trainer_interface_cleanup.md
agent_notes/loose_notes/2026-04-20_13-20-10_jsonc_config_migration.md
```

### Key Learnings

Put the dense memory bank in exactly this file:

```text
agent_notes/key_learnings.md
```

`key_learnings.md` is for the most surprising things we learned by trying and
failing. It is not a second journal and it is not a changelog. Add only
unexpected, high-signal lessons that changed our model of the project.

Rules for `key_learnings.md`:

- keep it under 200 lines
- recompress older bullets instead of letting it grow
- prefer dense bullets over prose
- include failures and surprises, not obvious facts
- if a point only records what happened, put it in `loose_notes/`
- if a point changes how future agents should reason, compress it into `key_learnings.md`

Use `research_notes/` for more curated research writeups, paper notes, or
durable conclusions. Use `agent_notes/loose_notes/` for raw chronology and
decision history.

## Config Style

Training hyperparameters should be defined once, in checked-in JSONC files under `src/train_configs/`.

- Do not add environment-variable fanout for every knob.
- Do not mirror full config defaults in large Python dictionaries.
- Do not add argparse blocks that duplicate the config schema.
- Shell scripts should choose a config file and call the trainer with that path.
- Python trainers should accept a config dict or config path, normalize only runtime concerns such as `Path` values, and fail loudly when required keys are missing.
- If a backward-compatible default is needed for an older config, apply it once during config load/normalization. Do not scatter `cfg.get("key", magic_number)` across model construction, logging, or train-loop code.
- Runtime code should read normalized configs with explicit keys. Repeated `.get(..., default)` at use sites is a smell unless the value is truly optional and `None` has semantic meaning.
- For status prints, prefer small dictionary/summary helpers that iterate over named keys. Do not hand-build long f-string chains that duplicate config defaults or quietly drift from the schema.

Keep code lean by passing config sections through warm paths instead of destructuring and rebuilding the same data repeatedly. When a boundary needs renamed constructor parameters, keep that mapping in one small factory function close to the boundary.

Prefer JSONC (`*.jsonc`) for train configs so experimental notes can live next to the values they explain.
