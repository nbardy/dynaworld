# Algorithm Evolution Reflection

Date: 2026-05-20

This note sharpens the role of `alpha_evolve/` after the 20-paper pass. The
goal is not normal repo cleanup. The goal is to evolve algorithmic code where
many plausible implementations exist and only executable evaluators can tell
which one is real.

## Current Belief

AlphaEvolve-style search is valuable here when the target has this shape:

```text
bounded algorithm surface
+ many plausible variants
+ fast correctness/parity gate
+ meaningful speed/quality metric
+ hidden promotion gate
+ archive of prior attempts and failures
```

It is not needed for deterministic plumbing such as "normalize a config" unless
that plumbing is a cheap harness used to validate the runner itself. The real
targets are kernels, math, objectives, schedulers, and selection algorithms.

## What We Are Actually Trying To Evolve

### Renderer And VJP Kernels

Good targets:

```text
tile binning strategy
active-tile scheduling threshold
packed row layout
feature-gradient reduction
target-area visual VJP
hidden64 / W^T colorizer kernel
backward accumulation order
sparse-vs-dense support selection
```

Why evolution helps:

```text
The implementation space is jagged. Small changes in layout, accumulation,
threadgroup shape, or support representation can flip speed by large factors,
and many variants pass tiny parity while failing larger workloads. A persistent
candidate database prevents us from rediscovering the same negative forks.
```

Hard gates:

```text
parity against Torch/reference fixture
no NaNs/nonfinite gradients
zero or bounded tile overflow
same target/loss/frame semantics
no hidden skip-gradient behavior
```

Fitness:

```text
backward_ms
total_step_ms
memory/resident cache size
quality after short smoke
quality after selected longer smoke for elites
changed LOC / complexity penalty
```

### Math And Objective Code

Good targets:

```text
feature/RGB/probe loss mixtures
support sampling strategies
target-grid weighting
generated-probe scores
metric compression functions
curriculum schedules
promotion score functions
```

Why evolution helps:

```text
The objective can be executable but not obvious. We can evaluate candidates by
short-run loss/PSNR/probe movement, but we need hidden gates to prevent metric
hacking, proxy overfit, and "wins" that come from dropping a hard term.
```

Hard gates:

```text
loss components all present
no target leakage
same train/eval split
same frame count and support contract
hidden quality gate not exposed as prompt text
```

### Algorithmic Schedulers

Good targets:

```text
same-view plus novel-view sampler policy
frame-window/microbatch selection
checkpoint promotion policy
renderer backend selection from workload shape
candidate parent/island selection
probe/test sampling policy
```

Why evolution helps:

```text
Schedulers are discrete algorithms with many reasonable heuristics. A one-shot
implementation tends to encode one hunch. An evolver can compare many hunches
under the same logged gate.
```

Hard gates:

```text
no target/input overlap leak
separate metric keys for each batch kind
same manifest semantics
same W&B/logging policy
reproducible config path
```

## First Real Algorithm Targets

Ranked by usefulness and evaluator readiness:

```text
1. STAR UVT target-area visual VJP kernel variants.
2. STAR UVT feature-gradient / W^T reduction variants.
3. Renderer backend selector for workload shape.
4. Same-view plus heldout novel-view scheduler policy.
5. Generated-probe / CodeT selector for candidate patches.
6. Gate4 owner/candidate record compression variants.
```

Config normalization, context packing, and result parsing are still useful as
bootstrapping tasks, but they should not be mistaken for the destination.

## Required Runner Shape

The runner should support two modes.

### Mode A: Offline Selector

Use when candidates or benchmark JSONs already exist.

```text
input:
    candidate/test matrix JSON

output:
    consensus sets
    selected candidates
    pass@k / n@k
    ranker gap
    false-positive rate
```

This is implemented first because it is cheap and can be tested without
spawning Codex.

### Mode B: Codex Candidate Evolution

Use when the runner should create new code variants.

```text
input:
    microlib task spec
    allowed paths
    evaluator commands
    prior candidates

loop:
    create candidate worktree
    call codex exec
    capture diff
    run evaluator cascade
    append candidate row
    update islands/elites
```

The first code in this folder should make Mode A real, then wire Mode B around
one target.

## Why AlphaEvolve Style Helps

Manual workflow today:

```text
think of one variant
implement it
run a gate
write a loose note
try another variant later
remember negative results by context
```

Evolved workflow:

```text
sample many variants
score each with the same evaluator
store every patch and failure class
cluster behavior
select elites
mutate from winners and informative failures
compare against oracle best-of-k
```

The key improvement is not autonomy. It is measurement:

```text
pass@k:
    did any generated candidate work?

n@k:
    did the selector choose a working candidate?

ranker_gap:
    did our selector miss a candidate that would have worked?

false_positive_rate:
    how often do visible/generated probes accept hidden-gate failures?

cost_per_solve:
    how much wall time, tokens, and evaluator time buy one promoted candidate?
```

## Failure Modes To Keep In Front

```text
benchmark-only win:
    candidate lowers runtime by skipping a real gradient or loss term

proxy overfit:
    candidate improves generated probes while failing hidden gates

largest-cluster trap:
    many wrong candidates share the same trivial behavior

dirty-tree leak:
    candidate depends on local untracked files or modified evaluators

scope creep:
    candidate rewrites the trainer instead of the kernel/helper target

unbounded cost:
    sampling more candidates hides the fact that the selector is bad
```

## Immediate Build Decision

Build the CodeT/AlphaCode selector core first. It gives us a reusable report
for any candidate pool and forces the right abstractions:

```text
candidate ids
visible gate result
generated-probe pass vector
hidden gate result
consensus set score
selected candidate ids
ranker gap
```

Then attach that selector to one real algorithmic target: STAR UVT target-area
visual VJP variants or renderer backend selection.
