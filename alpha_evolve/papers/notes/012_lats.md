# 012 - Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2310.04406
    https://arxiv.org/pdf/2310.04406
    https://github.com/lapisrocks/LanguageAgentTreeSearch
    https://lapisrocks.github.io/LanguageAgentTreeSearch/

Implementation artifacts inspected:
    https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/README.md
    https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/hotpot/lats.py
    https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/webshop/lats.py
    https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/programming/mcts.py
    https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/programming/main.py
    https://github.com/lapisrocks/LanguageAgentTreeSearch/blob/main/programming/generators/py_generate.py

Bibliographic metadata:
    Authors: Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang,
    Yu-Xiong Wang.
    First arXiv submission: 2023-10-06.
    Latest arXiv version inspected: v3, 2024-06-06.
    Venue/context: ICML 2024.

Why this paper matters for alpha_evolve:
    LATS is ToT plus ReAct plus Reflexion plus MCTS. It is the first paper in
    this queue that directly says how to use a tree search over action
    trajectories with external observations, value estimates, backpropagation,
    and reflection memory. For `alpha_evolve`, it is the bridge from "branch over
    patch plans" to "run a search process over actual candidate patches and
    evaluator feedback."

    The programming implementation is especially relevant: each action is a
    full code solution, visible generated tests provide observations, percentage
    of passed tests becomes reward, failed solutions generate reflections, and
    MCTS chooses where to expand next.

One-sentence mechanism:
    Use Monte Carlo Tree Search over language-agent states; expand with sampled
    LM actions, evaluate states with an LM/self-consistency value and external
    feedback, simulate to terminal rewards, backpropagate rewards through the
    tree, and store reflections from failed trajectories as future context.

## Reading Questions

- What is the executable feedback signal?
  Environment observations and terminal rewards. In HotPotQA, the environment
  provides search/lookup observations and correctness feedback. In programming,
  generated unit tests and compiler/test feedback are observations, while real
  test-suite success is final evaluation. In WebShop, browser observations and
  the final shopping reward/score are the signal.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Action trajectories. In programming, a trajectory can be shallow because each
  action is a complete solution candidate. In HotPotQA and WebShop, a trajectory
  is a ReAct-style sequence of thoughts/actions/observations.

- What is the population/database/selection mechanism?
  MCTS tree nodes with visits and values. Selection uses UCT. Expansion samples
  `n` actions. Simulation continues from a selected node until terminal or depth
  limit. Backpropagation updates ancestor values using returned reward.
  Reflection memory stores failed trajectories and natural-language feedback.

- What evidence proves the loop improves over one-shot generation?
  LATS outperforms ReAct, Reflexion, ToT, and RAP variants on HotPotQA,
  HumanEval, MBPP, WebShop, and Game of 24 in the reported experiments. Key
  numbers include HumanEval GPT-4 92.7 pass@1, HumanEval GPT-3.5 83.8 pass@1,
  MBPP GPT-3.5 81.1 pass@1, WebShop score 75.9 with GPT-3.5, and HotPotQA
  acting LATS 0.63 EM or 0.71 when combining CoT and ReAct.

- What does the method assume that DynaWorld does not have?
  It assumes state reversion is feasible. For pure text tasks this is easy:
  reset the prompt/context. For a repo, it means every candidate expansion must
  run in an isolated worktree or patch sandbox so the search can return to a
  parent state without mixing filesystem edits. It also assumes enough budget to
  expand many children.

## Mechanism

LATS defines a node as a state:

```text
s = [x, a_1 ... a_i, o_1 ... o_i]
```

where:

```text
x:
    original task/input

a_i:
    action or reasoning trace

o_i:
    observation returned by the environment
```

The paper lists six operations:

```text
selection
expansion
evaluation
simulation
backpropagation
reflection
```

### Selection

LATS selects a child using UCT:

```text
UCT(s) = V(s) + w * sqrt(log(N(parent)) / N(s))
```

This balances exploitation and exploration. A child with a high value is chosen
often, but less-visited children retain exploration pressure.

Local mapping:

```text
V(s):
    visible score, test pass fraction, LM value, risk-adjusted score

N(s):
    number of times candidate branch has been expanded/evaluated

w:
    exploration weight
```

For DynaWorld, `w` is not cosmetic. Too low and the runner gets stuck refining a
plausible but wrong patch family. Too high and it spends edits on noisy branches.

### Expansion

From a selected node, LATS samples `n` actions from the LM. In a ReAct task,
those are next thought/action steps. In programming, those can be complete code
solutions.

Repo code evidence:

```text
programming/mcts.py:
    for each selected node:
        generate n new solutions
        attach each as a child
        execute child on internal tests
```

Local mapping:

```text
candidate_expander:
    run codex exec to generate n candidates
```

But with a repo we cannot let `n` children edit the same worktree. Each child
needs a patch sandbox:

```text
base commit/tree
parent patch id
child patch id
apply in temp worktree
run visible evaluator
store diff and artifacts
discard worktree or keep as artifact
```

### Evaluation

LATS assigns scalar value to each child using:

```text
V(s) = lambda * LM(s) + (1 - lambda) * SC(s)
```

where:

```text
LM(s):
    language-model value estimate for expected success

SC(s):
    self-consistency score from repeated/similar action sampling
```

The key distinction from ToT is that LATS evaluates after environment feedback.
The value function can see observations, not just internal reasoning.

Local mapping:

```text
value = weighted_sum(
    deterministic visible score,
    generated-test pass fraction,
    LLM risk/value estimate,
    novelty/self-consistency,
    cost/diff penalties,
)
```

Do not rely on LM value alone for code. Use it as one input to a visible score.

### Simulation

From the selected/expanded node, LATS simulates forward until terminal state or
depth limit. In programming, the paper skips simulation because each action is a
complete solution; it directly uses test pass fraction as reward.

Local mapping:

```text
patch-plan state:
    simulation = generate patch, run visible evaluator

patch state:
    simulation = run repair attempts until pass/fail/depth

prompt state:
    simulation = run prompt on eval cases
```

### Backpropagation

When terminal reward `r` is known, LATS updates values along the path:

```text
N(s_i) += 1
V(s_i) = (old aggregate + r) / N(s_i)
```

The repo implementation accumulates reward on child and ancestor nodes. For
local `alpha_evolve`, this means a successful patch should also improve the
score of its patch plan and parent family. A failed branch should weaken its
ancestors enough that search explores siblings.

### Reflection

For failed terminal nodes, LATS prompts the LM with the failed trajectory and
final reward to generate self-reflection. It stores failed trajectories and
reflections, then feeds them into future action generation and value prompts.

This is Reflexion embedded inside MCTS.

Local mapping:

```text
failed child:
    store trace
    build reflection from visible evaluator feedback
    attach to node and maybe ancestor family
```

Important: reflection should be scoped to the branch. It should not become
global memory unless it proves useful across siblings or passes a utility gate.

## Implementation Notes From The Repo

### Programming MCTS

The programming implementation adapts Reflexion code. Important pieces:

```text
Node:
    solution
    parent
    children
    value
    visits
    depth
    reflection
    test_feedback

run_mcts:
    generate internal tests
    generate initial solution
    execute on internal tests
    reflect on failure
    for max_iters:
        select best child by UCT
        expand with n new solutions
        execute children on internal tests
        reflect on failed children
        reward = internal pass fraction + real pass reward
        backpropagate reward
    choose best solution
```

The implementation gathers context from parent nodes:

```text
accumulated_feedback
accumulated_reflection
```

This is a concrete design for local candidate prompt context:

```text
include feedback/reflections along current branch
do not dump every global failure
```

The code also regenerates internal tests during iterations. That may increase
test diversity, but it complicates scoring. Local runner should record the test
set fingerprint per value update.

### HotPotQA LATS

The HotPotQA implementation keeps global:

```text
reflection_map
failed_trajectories
```

It formats failed trajectories into ReAct-style traces and passes reflections to
both the generator and value prompts. The node stores:

```text
thought
action
observation
depth
reward
exact match
```

This is closer to a shell/code agent loop, where each node is:

```text
thought/plan
action/command-or-patch
observation/output
```

### WebShop LATS

WebShop introduces a practical issue for repo work: environment state must be
cloneable. The implementation has `clone_state()` over sessions, and the paper
explicitly notes that LATS assumes reverting to earlier states is feasible.

Local implication:

```text
repo LATS requires worktree snapshots or patch application rollback
```

Do not implement tree search by applying child patches in the main worktree and
trying to clean up with manual reset. Use isolated worktrees or patch files.

## Results

### HotPotQA

Reasoning-only GPT-3.5:

```text
Base LM:
    0.32 EM

CoT:
    0.34

CoT-SC:
    0.38

ToT:
    0.55

RAP:
    0.60

LATS (CoT):
    0.62
```

Acting-based GPT-3.5:

```text
ReAct:
    0.32 EM

ReAct best of k:
    0.38

Reflexion:
    0.51

ToT (ReAct):
    0.39

RAP (ReAct):
    0.54

LATS (ReAct):
    0.63

LATS n=10:
    0.65

LATS CoT + ReAct:
    0.71
```

The important detail is that naive ToT/ReAct adaptation performs poorly. Search
over action trajectories is not just ToT pasted onto ReAct; state, observations,
value functions, and backpropagation matter.

### Programming

HumanEval pass@1:

```text
GPT-3.5 CoT:
    46.9

GPT-3.5 ReAct:
    56.9

GPT-3.5 Reflexion:
    68.1

GPT-3.5 ToT:
    54.4

GPT-3.5 RAP:
    63.1

GPT-3.5 LATS:
    83.8

GPT-4 Base:
    80.1

GPT-4 Reflexion:
    91.0

GPT-4 LATS:
    92.7
```

MBPP pass@1:

```text
CoT:
    54.9

ReAct:
    67.0

Reflexion:
    70.0

ToT:
    65.8

RAP:
    71.4

LATS:
    81.1
```

Programming setup:

```text
generated tests:
    4 asserts per problem

expansion:
    sample 5 solutions

iterations:
    k = 8

reward:
    percentage of generated tests passed, plus real pass at terminal selection
```

This is the closest paper blueprint for a Codex-based microlib evolver.

### WebShop

WebShop score and success rate:

```text
ReAct:
    score 53.8, SR 28.0

ReAct best of k:
    score 59.1, SR 32.0

Reflexion:
    score 64.2, SR 35.0

LATS:
    score 75.9, SR 38.0

IL + RL baseline:
    score 62.4, SR 28.7

fine-tuning:
    score 67.5, SR 45.0
```

The paper observes that reflections in WebShop can be generic and local-minima
prone, echoing Reflexion. LATS helps by exploration, not by making reflection
magically correct.

### Game of 24

GPT-3.5 success:

```text
CoT:
    0.08

Reflexion:
    0.12

ToT:
    0.20

RAP:
    0.40

LATS:
    0.44
```

This shows LATS can also work in reasoning-only tasks, but its main value for
this repo is action-plus-observation search.

### Ablations

HotPotQA ablation:

```text
ToT (ReAct):
    0.39

RAP (ReAct):
    0.54

LATS no LM heuristic:
    0.37

LATS DFS:
    0.42

LATS no reflection:
    0.58

LATS full:
    0.63
```

Interpretation:

```text
LM value heuristic is crucial
MCTS beats DFS in this setting
reflection helps, but less than search/value
```

Cost table:

```text
k=10:
    ToT 0.34 / 33.97 nodes
    RAP 0.44 / 31.53 nodes
    LATS 0.44 / 28.42 nodes

k=30:
    ToT 0.39 / 47.54 nodes
    RAP 0.50 / 37.71 nodes
    LATS 0.52 / 34.12 nodes

k=50:
    ToT 0.49 / 84.05 nodes
    RAP 0.54 / 70.60 nodes
    LATS 0.61 / 66.65 nodes
```

LATS is still expensive, but better search can reduce nodes to success relative
to weaker tree methods.

## Design Implications For `alpha_evolve`

### LATS Requires Real State Isolation

The paper's reversion assumption becomes a hard implementation constraint:

```text
each node must be reproducible from:
    base commit
    parent patch chain
    local environment fingerprint
    evaluator fingerprint
```

Child expansions should run in separate worktrees or patch sandboxes. A simple
MCTS runner that edits the main repo in place will corrupt the tree.

### External Feedback Is The Reason To Use LATS

Do not run LATS on pure model preferences first. Use it where visible evaluator
feedback exists:

```text
generated tests
unit tests
smoke tests
benchmark slices
artifact validators
lint/schema checks
runtime metrics
```

Without external feedback, ToT/Self-Refine are cheaper and simpler.

### Backpropagate To Patch Plans And Families

If a patch generated from a plan succeeds, update:

```text
patch node
patch plan node
parent family
possibly prompt mutation rule
```

If it fails, weaken the same path and store the failure reflection. This is how
MCTS becomes useful for program search instead of just another retry loop.

### Separate Visible Reward From Hidden Fitness

LATS programming uses generated tests for reward and real tests for pass@1. For
DynaWorld:

```text
visible_reward:
    candidate-visible tests/smokes/metrics

promotion_fitness:
    repo-owned heldout gate, baseline table, user acceptance
```

Backpropagate visible reward during search. Promote only with hidden/repo-owned
fitness.

### Reflection Is Branch-Local

LATS stores failed trajectories and reflections. In local evolution:

```text
branch-local reflection:
    safe to feed to descendants

family-level reflection:
    only after repeated evidence

global reflection:
    only after human/strong gate confirmation
```

This avoids poisoning unrelated branches.

## Proposed Microlibs

### `mcts_node_store`

Responsibility:

```text
persist node ids, parent links, visits, values, rewards, patch refs, and
evaluator fingerprints
```

Fields:

```text
node_id
parent_id
state_kind
patch_ref
artifact_ref
depth
visits
value_sum
value_mean
terminal_reward
visible_reward
hidden_fitness_status
```

### `uct_selector`

Responsibility:

```text
choose child nodes using configurable UCT
```

Config:

```text
exploration_weight
min_visits_before_exploit
risk_penalty
novelty_bonus
```

### `sandboxed_expansion_runner`

Responsibility:

```text
expand a node into n isolated child worktrees/patches using codex exec
```

Invariant:

```text
no child writes directly into parent workspace
```

### `visible_reward_backpropagator`

Responsibility:

```text
update visits and values along a parent chain after visible evaluation
```

Inputs:

```text
node_id
reward
reward_components
evaluator_fingerprint
```

### `branch_reflection_memory`

Responsibility:

```text
attach reflections to failed trajectories with branch scope
```

Selection:

```text
descendants receive branch reflections
siblings receive only summarized family evidence
global prompts receive none by default
```

### `rollback_contract_checker`

Responsibility:

```text
verify a target problem can support LATS
```

Checks:

```text
can recreate parent state
can run candidate in temp worktree
can clean artifacts
can fingerprint evaluator
can replay selected candidate
```

If false, use linear repair or serial evolution instead.

## Local Falsification Tests

### Test 1: MCTS Beats Equal-Budget Reflexion

Target:
    one deterministic microlib coding task with generated tests and hidden
    tests.

Compare:

```text
Reflexion:
    1 initial + 7 repair attempts

LATS:
    k=8 iterations, n=5 expansion, same max Codex-call budget or normalized
    by executed candidate count
```

Expected:
    LATS solves more or reaches hidden pass with fewer executed candidates.

Failure:
    If Reflexion wins, tree overhead or value function is not justified.

### Test 2: UCT Selection Actually Changes Search

Procedure:

```text
run with exploration_weight = 0
run with default exploration_weight
run with high exploration_weight
```

Expected:
    Branch coverage and solve rate change. If not, values/visits are not wired
    correctly.

### Test 3: Branch-Local Reflection Avoids Poisoning

Procedure:

```text
make one branch fail with a misleading reflection
ensure unrelated sibling prompts do not receive that reflection
```

Expected:
    only descendants of the failed branch receive it.

### Test 4: Generated-Test Reward Does Not Promote

Procedure:

```text
candidate passes generated tests
candidate fails repo-owned hidden test
```

Expected:
    backpropagate visible reward if useful, but mark hidden promotion failed and
    prevent skill-library promotion.

### Test 5: Rollback Contract

Procedure:

```text
expand two sibling child patches
evaluate both
replay parent
expand third child
```

Expected:
    parent state is identical before each child expansion.

Failure:
    use of main worktree or leaked artifacts invalidates tree search.

## How It Changes The AlphaEvolve Plan

Before LATS:

```text
ToT over patch plans
LLaMEA serial mutation
program database / islands
```

After LATS:

```text
ToT over patch plans for cheap branching
LATS over patch candidates when external feedback exists and rollback is cheap
LLaMEA serial mutation for lower-cost parent evolution
AlphaEvolve-style islands/database once candidate persistence and promotion gates are stable
```

The choice is target-dependent:

```text
linear Reflexion:
    concrete failure, clear evaluator output

ToT:
    choose among plans before editing

LATS:
    action search with executable observations and rollback

LLaMEA/AlphaEvolve:
    longer-running candidate archive and population search
```

## Target Problems In This Repo

Good first LATS targets:

```text
small generated-test repair benchmark
alpha_evolve microlib implementation tasks
prompt-template evolution with deterministic eval cases
isolated code helper optimization with fast unit tests
```

Bad first LATS targets:

```text
long GPU training
large renderer kernel experiments
dirty shared worktree edits
anything without replayable evaluator fingerprints
```

Best initial implementation:

```text
mcts_node_store
sandboxed_expansion_runner
visible_reward_backpropagator
branch_reflection_memory
```

Then run it on a tiny synthetic coding suite before touching DynaWorld trainer or
renderer code.

## Open Questions

- How should Codex-call budget be normalized against LATS expansion count?
- What visible reward is dense enough for MCTS but not so proxy-like that it
  drives metric hacking?
- Should generated tests be regenerated per node or fixed per problem to keep
  value comparisons stable?
- Can LATS operate over patch plans first, then only materialize selected child
  patches?
- What branch-local reflection format avoids poisoning sibling branches?
- How much filesystem isolation is enough: git worktree, patch files, or full
  copy?

## Bottom Line

LATS is the strongest blueprint so far for a Codex-based search runner, but only
when three prerequisites hold: executable external feedback, branch rollback,
and enough budget for expansion. It should not be the first tool for every
problem. Use it after ToT-style plan branching identifies a target where
candidate patches can be evaluated quickly and replayed safely. Keep visible
reward, hidden promotion, and branch-local reflection separate.
