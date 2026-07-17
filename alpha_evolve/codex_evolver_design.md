# Codex-Driven Evolve Loop Design

This note maps AlphaEvolve's paper pattern onto this repo using Codex as the
code generator. The local loop should be boring infrastructure: create isolated
candidate worktrees, ask Codex to improve one microlib, run evaluators, save
patches and metrics, then sample the next prompt from the program database.

## Core Object

Each problem gets a microlib:

```text
alpha_evolve/microlibs/<problem>/
  problem.md              # human-readable contract
  context.md              # relevant repo files, current facts, non-goals
  prompt.md               # rendered into a Codex prompt
  evaluator.py            # returns JSON score dict
  allowed_paths.txt       # write scope
  forbidden_patterns.txt  # quick rejection rules
  seeds/                  # optional seed patches or baseline variants
```

The evolved object is normally a patch against repo code. For risky shader or
trainer lanes, the first evolved object can be an experiment-side helper or
prototype file under `research_experiments/` before promotion to `src/train/`.

## Candidate State

Use append-only candidate folders:

```text
outputs/alpha_evolve/<problem>/<run_id>/
  run_config.json
  programs.jsonl
  islands.json
  candidates/
    cand_000001/
      prompt.md
      codex_events.jsonl
      final.md
      patch.diff
      changed_files.txt
      eval_stage0.json
      eval_stage1.json
      eval_stage2.json
      stdout.log
      stderr.log
      status.txt
```

`programs.jsonl` is the evolutionary database. One row per candidate:

```json
{
  "candidate_id": "cand_000001",
  "parent_ids": ["seed"],
  "island": "fast_backward",
  "patch_path": "candidates/cand_000001/patch.diff",
  "metrics": {
    "correct": true,
    "parity_max_abs_error": 1.2e-6,
    "backward_ms": 421.7,
    "loss_final": 0.2929,
    "overflow_tile_count": 0
  },
  "notes": "short final-message summary",
  "accepted": true
}
```

## Runner Loop

1. Pick a problem and evaluator budget.
2. Create a clean candidate worktree from a known base commit or base patch.
3. Render a prompt from:
   - microlib contract
   - allowed paths
   - top prior candidates and their scores
   - a few diverse failed candidates and failure reasons
   - current evaluator target
4. Run Codex non-interactively in the candidate worktree.
5. Save the diff and reject if:
   - no relevant files changed
   - files outside `allowed_paths.txt` changed
   - forbidden commands/patterns appear
   - the diff edits docs only for an implementation problem
6. Run evaluator stages.
7. Insert the result into `programs.jsonl`.
8. Update island elites and sample the next parent.

## Codex Command

Preferred command:

```bash
codex exec \
  --cd "$WORKTREE" \
  --sandbox danger-full-access \
  --ask-for-approval never \
  --json \
  --output-last-message "$CANDIDATE/final.md" \
  "$PROMPT" > "$CANDIDATE/codex_events.jsonl"
```

Current CLI note: `codex -p` is profile, not prompt. If the desired outer UX is
`codex -p "..."`, make `alpha_evolve/bin/codex-p` a wrapper around the command
above.

## Isolation

Do not run evolution in the user's dirty working tree. This repo commonly has
many active uncommitted files. Candidate generation should use either:

- `git worktree add outputs/alpha_evolve/worktrees/<candidate> <base_ref>`
- a copied tree from a clean tarball snapshot
- a branch created from a known base commit after explicitly saving a base diff

The candidate worktree must be disposable. Evaluators can write outputs under
the candidate's `outputs/alpha_evolve/...` folder, not into shared benchmark
locations unless a candidate is promoted.

## Evaluation Cascade

Every microlib should define stages like this:

### Stage 0 - Static Rejection

Fast checks:

- patch applies cleanly
- allowed-path scope
- no destructive commands or broad deletes
- no W&B-disabling changes in benchmark candidates
- no config fanout or env-var fanout where JSONC config is the repo pattern
- `python -m py_compile` only as a syntax guard, never as final proof

### Stage 1 - Unit Or Parity

Cheap runtime:

- focused pytest node
- parity script with tiny tensors
- JSON schema check
- no nonfinite values
- exact contract flags present in JSON

### Stage 2 - Smoke Train Or Timing

Small real call graph:

- 1-step or 2-step trainer smoke
- 10 or 20-step mini overfit if the problem is optimization quality
- MPS timing if the lane is a renderer/shader target
- media/log outputs only when they are part of the checked contract

### Stage 3 - Main Comparison

Only for elites:

- current promoted config rerun
- W&B enabled for meaningful benchmarks
- `BASELINES.md` row only when this becomes a benchmark claim
- loose note on promotion or surprising failure

## Scoring

Use a score dict, not one scalar:

```json
{
  "correct": true,
  "scope_ok": true,
  "finite": true,
  "loss_delta": -0.026,
  "psnr_delta": 0.37,
  "backward_ms": 309.31,
  "total_step_ms": 1226.04,
  "overflow_tile_count": 0,
  "max_tile_load": 252,
  "changed_loc": 120,
  "complexity_penalty": 0.12
}
```

Selection should be Pareto-ish:

- correctness gates are hard filters
- primary score depends on the microlib
- complexity and changed file count are tie-breakers
- diversity bins preserve qualitatively different approaches

Example bins for STAR feature:

- `rgb_grad_handoff`
- `feature_grad_reduce`
- `fixedbin_backward`
- `support_pruning_or_schedule`
- `memory_valve`

## Prompt Sampling

A prompt should include:

- the exact problem contract
- non-goals and known negative results
- current best candidate metrics
- two or three diverse alternative candidates
- recent failure snippets
- allowed write paths
- evaluator command snippets
- explicit final response schema

Do not ask Codex to "be creative" without a hard patch target. Good prompt:

```text
Improve the STAR UVT F32 feature backward microlib. You may edit only these
paths. Keep parity with the tiny F4/F32 evaluator. The primary score is lower
64f/256px/32768t/F32 backward_ms with zero overflow and no loss regression.
Do not use skip-feature-gradient or disable feature gradients.
```

## Promotion

A candidate can become repo code only after:

- Stage 0-2 pass in a clean worktree.
- Its patch is reviewed against the repo style.
- The relevant smoke from `AGENTS.md` is run in the real repo if the change
  touches trainer signatures, config keys, dataclass fields, or return shapes.
- The docs/lane registry are updated if the result changes an active lane.

## Failure Modes To Guard

- Metric hacking: lower runtime by skipping gradients, changing targets, or
  silently dropping frames.
- Evaluator leakage: candidate edits the evaluator or baseline JSON.
- Dirty-tree contamination: candidate accidentally depends on local untracked
  files not included in the patch.
- Long-run overfitting: candidate wins a two-step smoke but destabilizes a
  20-step or promoted config.
- Scope creep: Codex rewrites the trainer instead of improving the microlib.
- W&B blindness: meaningful benchmark candidate disables logging.
- Prompt drift: later prompts forget repo-specific negative results.
