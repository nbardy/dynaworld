# Alpha Evolve Notes For Dynaworld

Date: 2026-05-18

This folder is a local design notebook and starter implementation for an
AlphaEvolve-inspired loop aimed at DynaWorld problems. The purpose is to turn
the paper pattern into concrete, repo-shaped microlibs that can be evolved by
non-interactive Codex runs and judged by hard evaluators.

## Sources Checked

- AlphaEvolve paper: https://arxiv.org/abs/2506.13131
- Google DeepMind launch post, 2025-05-14:
  https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/
- Google DeepMind impact post, 2026-05-07:
  https://deepmind.google/blog/alphaevolve-impact/

Key paper mechanics to copy locally:

- Put an automated evaluator at the center. LLM ideas are only useful when code
  execution produces objective scores.
- Evolve code, not prose. The paper uses marked evolve blocks and can edit
  whole files or larger codebases.
- Feed back prior programs, scores, logs, and rendered/evaluator output into
  later prompts.
- Use evaluation cascades: cheap correctness first, then timing/parity, then
  train/eval quality.
- Optimize multiple scores, not one scalar too early. Different metrics keep
  diverse implementation shapes alive.
- Keep a program database with diversity pressure. MAP-Elites and island-style
  populations matter because repeated local improvement collapses quickly.

## Repo Fit

DynaWorld is a good target only where the problem can be packaged as:

```text
seed code + allowed edit surface + evaluator cascade + score schema
```

Good targets here are narrow implementation or experiment-design bottlenecks
with existing JSON outputs, smokes, or parity tests:

- STAR UVT feature fast-backward / RGB-gradient handoff.
- Mixed same-view plus heldout-novel-view scheduler.
- Gaussian 512px promotion guards for the 300-clip V-JEPA lane.
- V-JEPA/F32 multicam benchmark-contract scripts.
- WorldFoam Gate4 owner/candidate record compression.
- Small code-organization helpers whose behavior is protected by smokes.

Bad targets for the first loop:

- "Invent the final world model."
- Long paid training runs without a fast local proxy.
- Any lane where the metric is a screenshot someone must inspect by hand.
- Broad trainer rewrites that cannot be checked in a candidate worktree.

## Codex Invocation Reality

The current local CLI reports:

```text
codex exec [OPTIONS] [PROMPT]
codex -p, --profile <CONFIG_PROFILE>
```

So `codex -p "..."` is not prompt input in this install. It selects a config
profile. The evolver should call:

```bash
codex exec --cd "$WORKTREE" --sandbox danger-full-access --ask-for-approval never \
  --output-last-message "$CANDIDATE/final.md" \
  "$PROMPT"
```

If we want the user-facing spelling `codex -p "..."`, implement a tiny wrapper
inside the evolver that accepts `-p/--prompt` and translates to `codex exec`.
Do not build the runner around a CLI flag that currently means profile.

## Folder Map

- `codex_evolver_design.md` - concrete architecture for the local runner.
- `algorithm_evolution_reflection.md` - why the real targets are kernels, math,
  objectives, schedulers, and selectors rather than normal cleanup.
- `experiment_backlog.md` - ordered experiments from offline selector sanity
  check to STAR/Gate4 algorithm evolution.
- `problem_targets.md` - ranked DynaWorld problems and evaluator sketches.
- `prompt_contract.md` - prompt shape, response contract, and guardrails.
- `evolver/` - stdlib Python starter code for candidate/probe selection reports
  and Codex command construction.
- `examples/` - runnable selector input examples.
- `microlibs/README.md` - reusable microlib anatomy.
- `microlibs/*.md` - first microlib specs for the problems we care about.

## First Implementation Bias

Start with one microlib and one evaluator, not a general framework. The best
first candidate is STAR UVT feature RGB-gradient handoff because:

- It has a measured bottleneck: feature-gradient/backward cost.
- It has existing parity and timing scripts.
- It can produce JSON metrics without a long run.
- It has a clear non-goal: do not promote benchmark-only skip-feature-gradient.

The mixed scheduler is the best second candidate because it is core to the
world-token contract, but its evaluator needs a real 1-step and 10-step trainer
smoke before evolution will be safe.

## Runnable Starter

The first usable code is the offline CodeT/AlphaCode selector. It scores an
already-executed candidate/probe matrix and reports consensus sets, selected
candidates, hidden-gate success if labels are present, ranker gap, and visible
false-positive rate.

```bash
PYTHONPATH=. uv run python -m alpha_evolve.evolver.cli \
  alpha_evolve/examples/codet_selector_matrix.json
```

This is intentionally smaller than a full runner. It lets us validate selector
logic before spending Codex calls or compiling kernel candidates.
