# 011 - Tree of Thoughts: Deliberate Problem Solving with Large Language Models

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2305.10601
    https://arxiv.org/pdf/2305.10601
    https://github.com/princeton-nlp/tree-of-thought-llm

Implementation artifacts inspected:
    https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/README.md
    https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/run.py
    https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/methods/bfs.py
    https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/tasks/game24.py
    https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/tasks/text.py
    https://github.com/princeton-nlp/tree-of-thought-llm/blob/master/src/tot/tasks/crosswords.py

Bibliographic metadata:
    Authors: Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran,
    Thomas L. Griffiths, Yuan Cao, Karthik Narasimhan.
    First arXiv submission: 2023-05-17.
    Latest arXiv version inspected: v2, 2023-12-03.
    Venue/context: NeurIPS 2023 camera-ready.

Why this paper matters for alpha_evolve:
    Tree of Thoughts is the cleanest paper so far for separating four choices
    that an AlphaEvolve-style Codex runner must make:

```text
1. What is the state?
2. How do we expand a state?
3. How do we evaluate partial states?
4. What search policy explores the tree?
```

    ReAct and Self-Refine are mostly linear. Reflexion adds memory to a retry
    chain. ToT is the first explicit tree-search paper in the queue. For
    DynaWorld, this maps to candidate families: a "thought" is not hidden
    chain-of-thought, but a durable partial artifact such as a design sketch,
    prompt variant, patch plan, generated test suite, loss-shaper candidate, or
    small microlib implementation. Search can branch over those artifacts, score
    them with cheap visible evaluators, then choose what to run next.

One-sentence mechanism:
    Decompose problem solving into intermediate "thought" states, generate
    multiple next thoughts from each state, heuristically evaluate the partial
    states, and use a search algorithm such as BFS or DFS to keep, prune, and
    backtrack among promising branches.

## Reading Questions

- What is the executable feedback signal?
  Mostly model-generated heuristic values or votes, plus task-specific final
  success checks. Game of 24 has a deterministic final checker. Creative Writing
  uses GPT-4 and human preference for coherence. Mini Crosswords uses letter,
  word, and game-level correctness, while state pruning uses LM judgments about
  whether remaining clues are possible.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Partial reasoning states. In this repo's terms, search should be over
  candidate artifacts and their measured state signatures, not private thoughts.
  A node can be a prompt plan, patch plan, generated test batch, microlib draft,
  or patch state.

- What is the population/database/selection mechanism?
  ToT keeps a frontier of states. BFS keeps the top `b` states at each depth.
  DFS explores the most promising state until a solution or prune threshold,
  then backtracks. This is not evolutionary selection over a long-lived program
  database, but it is a direct ancestor of archive/frontier management.

- What evidence proves the loop improves over one-shot generation?
  On Game of 24, GPT-4 with CoT solves 4 percent, CoT self-consistency solves
  9 percent, best-of-100 CoT solves 49 percent, ToT with breadth 1 solves 45
  percent, and ToT with breadth 5 solves 74 percent. On Creative Writing, ToT
  improves GPT-4 coherence scores and is preferred by humans over CoT more often
  than the reverse. On Mini Crosswords, ToT reaches 60 percent word-level
  success and solves 4 of 20 games, while IO/CoT stay below 16 percent word
  success.

- What does the method assume that DynaWorld does not have?
  It assumes a cheap and reasonably faithful partial-state heuristic. DynaWorld
  partial patches can be expensive to evaluate and may have deceptive metrics:
  a patch can improve a local smoke while violating the data contract, or look
  promising by shrinking work rather than solving the problem.

## Mechanism

ToT defines a state:

```text
s = [x, z_1, ..., z_i]
```

where `x` is the input and each `z_i` is an intermediate thought. The paper's
central abstraction is that a thought should be:

```text
small enough:
    the LM can generate diverse candidates

large enough:
    the LM/evaluator can judge whether the partial state is promising
```

That sentence is directly applicable to `alpha_evolve`. A mutation step should
not be:

```text
one token
```

because it is too small to evaluate meaningfully. It also should not be:

```text
rewrite the whole trainer
```

because it is too large to branch and recover from. Good local thought units:

```text
one microlib API contract
one evaluator stage
one patch strategy
one generated-test suite
one loss-shaper variant
one benchmark harness variant
one prompt mutation
```

The paper says an instantiation of ToT answers four questions.

### 1. Thought Decomposition

Examples from the paper:

```text
Game of 24:
    one intermediate equation

Creative Writing:
    one paragraph-level writing plan

Mini Crosswords:
    one word fill for one clue
```

DynaWorld examples:

```text
renderer optimization:
    one proposed bottleneck and one scoped kernel/helper change

training config cleanup:
    one normalized config boundary

alpha_evolve runner:
    one microlib plus its test contract

prompt evolution:
    one prompt section or mutation rule

generated tests:
    one batch of tests tagged by behavior class
```

The local search unit must include a way to roll forward and a way to score.

### 2. Thought Generation

The paper uses two strategies:

```text
sample:
    independently sample candidate thoughts from a CoT-style prompt

propose:
    ask the model to list several next thoughts in one prompt
```

Use `sample` when the space is rich and diverse:

```text
different algorithm sketches
different prompt rewrites
different problem framings
```

Use `propose` when the space is constrained and duplicate avoidance matters:

```text
which evaluator stage to add next
which failed metric to target
which file boundary to patch
which generated test category is missing
```

For Codex evolution, `propose` is often cheaper:

```text
codex exec "list 5 patch strategies, no edits"
```

then evaluate the strategies before spending full edit attempts. But once a
strategy is selected, candidate patch generation should usually run in a clean
worktree state.

### 3. State Evaluation

The paper evaluates states in two ways:

```text
value:
    score each state independently

vote:
    compare states and choose the most promising
```

Game of 24 uses values:

```text
sure
likely
impossible
```

Creative Writing uses votes because coherence is hard to score in isolation.
The repo's `bfs.py` implements both:

```python
get_values(task, x, ys, n_evaluate_sample)
get_votes(task, x, ys, n_evaluate_sample)
```

Local mapping:

```text
value:
    cheap deterministic score exists
    examples: import pass, unit test count, smoke metric, runtime delta,
    diff size, shape-contract flags

vote:
    relative quality matters more than absolute score
    examples: prompt clarity, plan plausibility, risk tradeoff, patch strategy
```

Important: ToT's value/vote heuristics are approximate. The paper explicitly
allows imperfect state evaluation; it only needs to be useful enough for search.
In DynaWorld, approximate visible heuristics must be tagged as shaping signals,
not promotion truth.

### 4. Search Algorithm

The paper uses BFS and DFS.

BFS:

```text
for each depth:
    expand all frontier states
    evaluate expanded states
    keep top b states
```

DFS:

```text
explore promising state
if solved:
    record output
if evaluator says subtree impossible:
    prune
else:
    continue deeper
on failure:
    backtrack to parent and try next state
```

The repo's `src/tot/methods/bfs.py` is deliberately small:

```text
ys = ['']
for step in task.steps:
    new_ys = expand ys
    values = value/vote new_ys
    select_new_ys = top n_select_sample
    ys = select_new_ys
return ys
```

That smallness is useful. The first local `alpha_evolve` tree-search runner
should be similarly plain:

```text
frontier = [initial_state]
for depth in range(max_depth):
    candidates = expand(frontier)
    feedback = evaluate_visible(candidates)
    frontier = select(candidates, feedback, breadth)
return best_visible(frontier)
```

Do not start with MCTS. Start with logged BFS over small artifacts.

## Experiments

### Game of 24

Task:

```text
Given four numbers, use +, -, *, / to produce 24 exactly.
```

Thought unit:

```text
one arithmetic step, leaving fewer numbers
```

Generation:

```text
propose possible next equations
```

Evaluation:

```text
classify remaining numbers as sure, likely, or impossible to reach 24
```

Search:

```text
BFS with breadth b = 5
```

Results:

```text
IO:
    7.3 percent

CoT:
    4.0 percent

CoT-SC with k=100:
    9.0 percent

IO + Refine, k=10:
    27 percent

IO best of 100:
    33 percent

CoT best of 100:
    49 percent

ToT b=1:
    45 percent

ToT b=5:
    74 percent
```

Critical lesson:

```text
structured branching beats independent whole-solution sampling
```

The paper's error analysis says around 60 percent of CoT samples already fail
after the first step. ToT's advantage is catching bad first steps before they
consume the whole trajectory.

DynaWorld mapping:

```text
bad first patch assumptions should be pruned early
```

Examples:

```text
"renderer bottleneck is attention"
"config alias refactor will reduce complexity"
"generated tests are sufficient"
"this metric is a proxy for heldout quality"
```

If a cheap visible probe can reject the assumption before a full patch, tree
search saves time.

### Creative Writing

Task:

```text
write a coherent four-paragraph passage ending each paragraph with a supplied
random sentence
```

Thought unit:

```text
a short writing plan
```

Generation:

```text
sample 5 plans, vote for the best
sample 5 passages from selected plan, vote for the best
```

Evaluation:

```text
GPT-4 scalar coherence score and human pairwise preference
```

Results:

```text
GPT-4 coherence:
    IO 6.19
    CoT 6.93
    ToT 7.56

human pairwise:
    ToT preferred over CoT in 41 of 100
    CoT preferred over ToT in 21 of 100
    38 similar
```

Iterative refine is competitive on this natural-language task, improving IO and
ToT coherence further. The paper suggests refinement can be a third thought
generation strategy: new thoughts arise by refining old thoughts.

DynaWorld mapping:

```text
plan-vote is good for soft artifacts
```

Use ToT vote for:

```text
which prompt mutation is clearer
which patch plan is lower risk
which note/synthesis framing is more useful
which generated-test suite is more behaviorally diverse
```

Do not let vote choose promotion.

### Mini Crosswords

Task:

```text
fill a 5x5 crossword with 5 horizontal and 5 vertical clues
```

Thought unit:

```text
one word fill for one clue
```

Search:

```text
DFS with backtracking
```

Evaluation:

```text
LM proposal confidence
LM state evaluation of remaining clue feasibility
letter, word, and game-level final correctness
```

Results:

```text
IO/CoT:
    less than 16 percent word-level success

ToT:
    60 percent word-level success
    4 of 20 games solved

ToT + oracle best state:
    7 of 20 games solved

no backtracking:
    word-level success only 20 percent
```

The most important result is not just that ToT improves. It is that pruning can
be wrong. The paper observes cases where the state evaluator prunes a solved or
promising state because rare words look impossible to GPT-4. Removing pruning
can sometimes find correct solutions that pruning misses, but may output the
wrong state by heuristic.

DynaWorld mapping:

```text
pruning is dangerous when heuristics are brittle
```

Never permanently discard a candidate just because an LLM judge dislikes it.
Archive it, mark it pruned for current budget, and keep enough metadata to
resurrect it if later evidence contradicts the heuristic.

## Implementation Notes From The Repo

### BFS Runner

The official BFS code is compact and modular:

```text
get_samples:
    sample complete/partial continuations

get_proposals:
    propose next steps in a constrained context

get_values:
    evaluate each candidate independently

get_votes:
    compare candidates as choices

solve:
    expand, evaluate, select, log, repeat
```

It also caches value prompts:

```text
task.value_cache[value_prompt] = value
```

Local version should cache:

```text
prompt hash
candidate diff hash
evaluator fingerprint
visible score
reflection/actionability score
```

### Game24 Task

`Game24Task` has a clean `test_output` that uses `sympy` to verify final
correctness. It also converts LM labels into numeric values:

```text
impossible -> 0.001
likely -> 1
sure -> 20
```

This is an ad hoc but explicit heuristic map. DynaWorld should copy the
explicitness, not the numbers:

```text
gate_fail -> 0
gate_unknown -> small positive
gate_pass -> high
metric_regression -> penalty
diff_risk -> penalty
```

### Text Task

`TextTask` uses GPT-4 scoring for coherence and a vote prompt to compare
candidate plans/passages. This maps to soft candidate plan selection.

### Crosswords Task

`MiniCrosswordsTask` maintains an environment-like board, status flags, proposal
caches, and feasibility checks. That is closer to a codebase search environment
than Game24:

```text
state:
    board + filled clues + changed clues

proposal:
    next word and confidence

evaluation:
    feasibility of remaining constraints
```

Local analog:

```text
state:
    worktree patch + evaluator results + file ownership + problem contract

proposal:
    next patch operation or candidate branch

evaluation:
    feasibility of reaching promotion under remaining budget
```

## Design Implications For `alpha_evolve`

### Use ToT For Branching Before Full Patches

Codex edits are expensive. Before generating five full patches, ask for five
patch strategies and evaluate them cheaply:

```text
generate patch plans
score plans by constraints, likely evaluator impact, risk, and novelty
select top b
generate full patches only for selected plans
```

This is ToT at a higher semantic level. It reduces wasted file churn.

### Separate Heuristic From Fitness

ToT value/vote is a heuristic. It guides search. It is not final truth.

DynaWorld split:

```text
heuristic_score:
    candidate-visible search guidance

visible_eval:
    cheap deterministic gates

hidden_fitness:
    repo-owned promotion gate
```

The program database should store all three separately.

### Backtracking Beats Linear Repair When Early Choices Matter

Use linear repair when:

```text
the candidate is close
the evaluator error is concrete
the next change is obvious
```

Use ToT-style branching when:

```text
the first assumption might be wrong
there are multiple plausible mechanisms
the target requires choosing a family of solution
the evaluator is costly enough that bad first choices are expensive
```

Renderer and trainer bottleneck work often falls into the second category.

### Candidate State Is Not Private Reasoning

Do not store hidden chain-of-thought. Store auditable state:

```json
{
  "state_id": "...",
  "parent_id": "...",
  "depth": 2,
  "artifact_kind": "patch_plan",
  "artifact_ref": "...",
  "visible_metrics": {},
  "heuristic_scores": {},
  "risk_flags": [],
  "selection_reason": "..."
}
```

This preserves ToT's search benefits without depending on private reasoning
text.

### Use Multiple Search Modes

Map paper search modes to local modes:

```text
BFS:
    breadth over patch plans or microlib variants

DFS:
    deep repair of one candidate with backtracking

beam search:
    keep top b candidates by visible score

sample:
    independent Codex proposals when diversity matters

propose:
    one Codex call lists constrained next actions

vote:
    relative plan/prompt/test quality

value:
    deterministic or scalar visible metrics
```

Do not force every target into BFS. The useful piece is the shared state,
expansion, evaluation, selection interface.

## Proposed Microlibs

### `thought_state_schema`

Responsibility:

```text
define auditable node records for ToT-style search over artifacts
```

Fields:

```text
state_id
parent_id
depth
problem_id
artifact_kind
artifact_ref
evaluator_fingerprint
visible_score
heuristic_score
risk_flags
selection_status
```

### `candidate_expander`

Responsibility:

```text
generate next candidate states from one or more parent states
```

Modes:

```text
sample:
    independent Codex proposals

propose:
    one Codex call lists multiple candidate next actions

refine:
    improve an existing candidate with feedback

mutate:
    LLaMEA/AlphaEvolve-style parent mutation
```

### `state_heuristic_evaluator`

Responsibility:

```text
score partial candidate states for search guidance
```

Inputs:

```text
artifact summary
visible evaluator facts
reflection memory
target contract
budget remaining
```

Outputs:

```text
value score
vote rank
risk flags
prune recommendation
confidence
```

Recommendation is not deletion.

### `frontier_selector`

Responsibility:

```text
select states to continue under a budget
```

Policies:

```text
greedy top b
score-proportional sample
pareto diversity
epsilon novelty
manual pin
```

### `backtracking_controller`

Responsibility:

```text
switch from current branch to sibling/ancestor when the branch stalls
```

Triggers:

```text
same failure repeats
score delta below threshold
budget cap reached
heuristic confidence drops
new evaluator fact invalidates ancestor assumption
```

### `prune_archive`

Responsibility:

```text
record pruned states and why, without deleting them
```

Fields:

```text
pruned_by
reason
confidence
evaluator facts available
resurrection_conditions
```

This copies the crossword lesson: pruning heuristics can be wrong.

## Local Falsification Tests

### Test 1: Plan Tree Beats Patch Sampling

Compare under equal Codex-call budget:

```text
A: generate 5 full patches independently
B: generate 5 patch plans, score/vote, generate full patches for top 2
```

Expected:
    B has better solve rate per edit or lower file churn.

Failure:
    Plan-level heuristic is too weak; direct patch sampling wins.

### Test 2: BFS Beats Linear Repair Only On Branchy Problems

Choose two target classes:

```text
concrete bug with clear traceback
ambiguous performance/architecture problem
```

Expected:
    Linear repair matches or beats BFS on concrete bug. BFS helps on ambiguous
    target.

Failure:
    If BFS does not help ambiguous targets, the state decomposition is wrong.

### Test 3: Pruned Candidates Can Be Resurrected

Procedure:

```text
mark a candidate pruned by heuristic
later run a deterministic evaluator that contradicts the prune reason
```

Expected:
    candidate can reenter frontier or trigger heuristic correction.

Failure:
    pruned states are effectively deleted.

### Test 4: Value And Vote Are Different

Run both on a soft artifact target:

```text
prompt template candidates
generated test-suite candidates
patch plan candidates
```

Expected:
    vote works better when absolute scoring is vague; value works better when
    deterministic metrics exist.

### Test 5: Thought Unit Size Sweep

Compare state units:

```text
one file-level patch plan
one function-level patch plan
one full implementation patch
one test-suite variant
```

Expected:
    too-small units are hard to evaluate; too-large units waste edits.

## How It Changes The AlphaEvolve Plan

ToT adds a planning/search layer before and around evolutionary mutation:

```text
one-shot codex exec
react_repair_loop + reflection
self_refine_loop for soft artifacts
ToT-style branch over plans/states
LLaMEA serial parent mutation
program database / islands
```

The new distinction:

```text
repair loop:
    fix one candidate

tree search:
    choose among candidate families or next actions

evolution:
    maintain and mutate a long-lived archive/population
```

A minimal implementation can share the same node schema across all three.

## Target Problems In This Repo

Good first ToT targets:

```text
choosing alpha_evolve microlib order
choosing generated-test categories
choosing patch plans for a small performance helper
choosing evaluator cascade design
choosing prompt mutation strategies
```

Bad first ToT targets:

```text
large trainer refactor
kernel-level renderer changes with slow feedback
anything requiring hidden W&B or paid GPU validation
```

The best initial local experiment:

```text
Use ToT over patch plans, not patches.
Then let codex exec implement only selected plans.
```

## Open Questions

- What is the right state unit for code evolution: patch plan, file diff, test
  suite, prompt section, or metric signature?
- Can a cheap LLM heuristic predict which patch plans pass deterministic gates?
- How often should a pruned state be resurrected or rechecked?
- Should ToT frontier selection include diversity descriptors, or leave that to
  later island evolution?
- Can reflection memory improve state evaluation, or will it make pruning more
  biased?
- What is the budget crossover where ToT beats independent sampling for Codex?

## Bottom Line

Tree of Thoughts is not just "ask the model to think harder." It is a modular
search interface: state, expand, evaluate, select, backtrack. For
`alpha_evolve`, the direct move is a small frontier runner over auditable
candidate artifacts. Use it first for plan selection and evaluator design. Keep
heuristics separate from fitness, archive pruned states, and only scale to
program-database evolution after the small tree beats equal-budget independent
Codex sampling.
