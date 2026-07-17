# 006 - Eureka: Human-Level Reward Design via Coding Large Language Models

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2310.12931
    https://arxiv.org/pdf/2310.12931
    https://eureka-research.github.io/
    https://github.com/eureka-research/Eureka

Bibliographic metadata:
    Authors: Yecheng Jason Ma, William Liang, Guanzhi Wang, De-An Huang,
    Osbert Bastani, Dinesh Jayaraman, Yuke Zhu, Linxi Fan, Anima Anandkumar.
    First arXiv submission: 2023-10-19.
    Latest arXiv version inspected: v2, 2024-04-30.
    Venue: ICLR 2024.

Why this paper matters for alpha_evolve:
    Eureka is the clearest paper so far on evolving executable feedback code
    while keeping a separate, external fitness function. It is not evolving the
    policy directly. It evolves reward functions, trains a policy under each
    reward, then selects by an external task metric. That separation is the
    exact guardrail DynaWorld needs if Codex ever evolves scoring helpers,
    shaped losses, curricula, tests, or diagnostic metrics.

One-sentence mechanism:
    Give the LLM pruned environment code and a task description, sample reward
    functions, train policies under those rewards, summarize policy/reward
    component traces as reward reflection, then mutate the best reward code.

## Reading Questions

- What is the executable feedback signal?
  The immediate generated artifact is reward code. The selection feedback is the
  task fitness of a policy trained using that reward, plus reward reflection:
  named reward-component traces, task metrics, success rates, episode lengths,
  and execution errors.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Free-form reward code. The learned policy is a downstream product of the
  reward and RL trainer, not the search artifact stored in the LLM context.

- What is the population/database/selection mechanism?
  Eureka samples a batch of K rewards per iteration, evaluates all executable
  samples, selects the best reward by task metric, and mutates that reward in
  the next iteration using its reflection. The official experiments use multiple
  random restarts, 5 iterations per run, and K = 16 samples per iteration.

- What evidence proves the loop improves over one-shot generation?
  The paper includes an ablation that spends the same number of reward samples
  on initial generation instead of iterative improvement. Iterative Eureka beats
  the wider one-shot sample baseline after two iterations, and reward reflection
  removal degrades performance substantially on Isaac tasks.

- What does the method assume that DynaWorld does not have?
  It assumes massively parallel GPU simulation where each candidate can be
  trained and judged in a reasonable wall-clock window. DynaWorld has slower
  training, heavier repo state, and stricter leakage concerns. We need cheaper
  proxy gates and stricter editable boundaries.

## Mechanism

Eureka formalizes reward design as a search over reward functions. The setup is:

```text
world/source code M
reward space R
learning algorithm A_M(R) -> policy pi
fitness function F(pi) -> scalar task score
```

The generated reward is not the ground-truth fitness. That distinction is the
paper's most important export to `alpha_evolve`.

In Eureka:

- the LLM writes reward function code R;
- the RL system trains a policy using R;
- the trained policy is evaluated with F;
- the best R is selected and shown back to the LLM;
- reward reflection explains how R behaved during training.

This makes the generated code a shaping signal, not the final judge.

### Environment As Context

Instead of hand-written task prompts and templates, Eureka feeds the LLM the
environment source code plus a natural-language task description. The paper
argues that the environment code tells the LLM what variables are available and
how they relate to the task.

The key engineering detail is pruning. The implementation does not dump the
entire simulator. It extracts the observation-relevant portion of the
environment code so the context fits and does not expose irrelevant internals.
For a new environment, the README flow asks users to verify standard RL first,
create an environment config, then run a pruning utility that constructs the raw
environment context and skeleton code for reward insertion.

DynaWorld mapping:

```text
Do not paste the whole repo into codex exec.
Give the candidate the smallest source slice that defines the editable contract.
Expose allowed state variables and helper APIs.
Hide final evaluator implementation and forbidden data paths.
```

This is directly relevant to same-view versus heldout-view leakage. If a
candidate prompt exposes the wrong state or labels, the generated code may
legitimately use them because they were in context.

### Initial Generation

The initial prompt asks for TorchScript-compatible reward code and requires
that the reward return:

```text
total_reward
dictionary_of_reward_components
```

The component dictionary is not cosmetic. It is the substrate for reward
reflection. If the generated reward has named components, the runner can log
their values through training and tell the LLM which component is stagnant,
overscaled, or correlated with success.

For DynaWorld, any evolved scoring/shaping helper should expose components:

```text
total_score
components = {
    "finite_gate": ...,
    "leak_penalty": ...,
    "quality_proxy": ...,
    "speed_penalty": ...,
    "loc_penalty": ...,
}
```

Even if final selection uses one scalar, the component names give reflection
enough handles to edit the artifact.

### Evolutionary Reward Search

Eureka samples multiple independent rewards per iteration. This is partly a
quality-diversity move and partly an error-recovery move: if some generated
reward functions fail to parse or run, a batch of 16 usually still contains at
least one executable candidate.

The paper's loop:

```text
for iteration in N:
    sample K reward functions from LLM
    insert each reward into environment skeleton
    train one RL policy per reward
    compute task fitness and reward-component traces
    select best reward in this iteration
    append best reward and reflection to prompt
return global best reward
```

The official code reflects this shape:

- load task/environment config with Hydra;
- load prompt fragments from `eureka/utils/prompts`;
- call OpenAI ChatCompletion with `n=chunk_size`;
- extract Python code blocks with regex;
- parse the reward function signature;
- inject the reward call into a generated Isaac Gym task file;
- launch `train.py` with the generated task;
- read TensorBoard logs;
- assemble feedback from reward-component metrics and tracebacks;
- select the best response by success;
- keep only the best assistant response and user feedback in the next prompt.

The last point matters: Eureka's prompt history is Markovian. It keeps the last
best reward and its reflection, not the whole lineage. That is more aggressive
compression than CodeEvolve and AlphaEvolve, but it works for this domain.

### Reward Reflection

Reward reflection is the core feedback mechanism. It converts training dynamics
into textual instructions for the next reward mutation.

The reflection includes:

- task score or success rate over checkpoints;
- episode length;
- values of each reward component over checkpoints;
- max, mean, and min for each tracked component;
- execution tracebacks when code fails;
- instructions about how to interpret common patterns.

The prompt tells the LLM to reason about components before writing new code. The
tips are operational:

- if success is near zero, rewrite the reward;
- if a component is nearly constant, rescale, rewrite, or discard it;
- if one component dominates magnitude, rescale it.

This is more useful than saying "score was bad." It gives credit assignment for
the generated feedback code.

The ablation is important: removing reward reflection and keeping only task
metric snapshots reduces average normalized score over Isaac tasks by 28.6
percent in the reported experiment. The degradation is worse on
high-dimensional tasks. That makes reflection a required mechanism, not a nice
commentary layer.

### Human Feedback

Eureka supports two human-feedback routes:

1. Human initialization.

   Replace the first generated reward with a human-written reward, evaluate it,
   then let Eureka mutate it from reflection. The paper reports that this
   improves over both pure Eureka and the original human reward on the tested
   tasks.

2. Human textual reflection.

   Replace automated reward reflection with human natural-language feedback
   about rollout behavior. In the humanoid running example, the generated reward
   becomes slower but more human-preferred because it optimizes a behavior
   quality not captured by forward velocity.

This is a major point for DynaWorld. Some failures are visible in generated
videos or artifacts before they are visible in a scalar. The user should be able
to add a reflection like:

```text
The heldout view is numerically improving, but the video looks smeared and the
object identity drifts. Penalize that failure without using heldout labels.
```

That reflection should steer prompt generation. It must not rewrite the hidden
acceptance metric.

## Evaluation

Eureka evaluates on 29 open-source RL environments across 10 robot morphologies:
9 Isaac Gym tasks and 20 Bidexterous Manipulation tasks. The paper emphasizes
that these environments were released at or after GPT-4's cited knowledge
cutoff, reducing the chance that GPT-4 memorized environment-specific reward
solutions.

Baselines:

- Human: original expert-shaped rewards from the benchmark tasks.
- Sparse: ground-truth task fitness functions.
- L2R: a prior LLM reward-design method using task-specific templates and API
  primitives.
- Eureka without evolution: wider first-iteration sampling instead of iterative
  mutation.
- Eureka without reward reflection.
- GPT-3.5 variant for model-quality sensitivity.

Training details:

- final reward functions are optimized with the same PPO implementation and
  task hyperparameters;
- final rewards get 5 independent PPO runs;
- performance uses the average of maximum task metric values from fixed
  checkpoint intervals;
- intermediate Eureka candidates are evaluated with one PPO run;
- official README defaults expose `iteration`, `sample`, `max_iterations`, and
  `num_eval` knobs.

Headline results:

- Eureka outperforms expert human rewards on 83 percent of tasks.
- It reports an average normalized improvement of 52 percent.
- It exceeds or matches human level on all Isaac tasks and 15 of 20 Dexterity
  tasks.
- Iterative evolution improves over wider one-shot sampling.
- Reward reflection is a large positive contributor.
- GPT-3.5 reduces performance but preserves much of the method's usefulness.

The most relevant result is not the exact robotics score. It is that a generated
white-box shaping signal can be better than expert hand-shaped rewards while a
separate task metric remains the judge.

### What The Results Actually Prove

The paper proves that LLM-generated reward code can be improved by evolutionary
search when:

- the environment state is exposed in code;
- candidate rewards are executable;
- reward components are logged;
- policy training is fast enough to evaluate many candidates;
- an external task metric exists;
- the prompt gets reflection, not only a final score.

It does not prove:

- that the generated reward is safe outside simulation;
- that optimizing a proxy reward avoids reward hacking;
- that no hidden simulator assumption leaks through context;
- that the approach works when each candidate is expensive to train;
- that natural-language feedback is always consistent or sufficient;
- that final task metrics capture all user intent.

For DynaWorld, the proof boundary is valuable. Eureka is a recipe for evolving
shaped training signals or diagnostic helper code, not a license to let Codex
edit the final evaluator.

## Failure Modes

### Reward Hacking By Editable Metrics

Eureka's central safety feature is that the generated reward R is separate from
the task fitness F. If a local system lets Codex edit both the shaping score and
the selection score, the method collapses into metric hacking.

DynaWorld rule:

```text
Generated code may shape training.
Generated code may summarize diagnostics.
Generated code may propose tests.
Generated code must not edit hidden acceptance.
```

### Context Leakage

Environment-as-context works because reward code needs environment variables.
But if context includes privileged labels, hidden heldout paths, or final
evaluator internals, the generated code can exploit them.

DynaWorld rule:

```text
Context pruning is part of the evaluator contract.
Allowed state and forbidden state must be explicit.
The prompt packer should be audited like production code.
```

### Expensive Candidate Evaluation

Eureka spends real RL training per candidate. The official experiments use
GPU-accelerated Isaac Gym and still report each independent run taking less than
one day on an 8 A100 station. This is not a cheap local loop.

DynaWorld rule:

```text
Use Eureka-style evolution only where a candidate can be judged by a cheap
proxy first. Reserve expensive train/eval for survivors.
```

### Reflection Can Encode The Wrong Goal

Automated reflection summarizes measurable components. Human reflection
summarizes subjective behavior. Both can be wrong or incomplete. The humanoid
example intentionally trades speed for preferred gait; that is good only
because the goal changed.

DynaWorld rule:

```text
Reflection text is prompt evidence, not ground truth.
Candidate selection still needs immutable metrics and heldout checks.
```

### Component Names Shape The Search

Reward components are editable handles. Poorly named or misleading components
can push the LLM toward bad surgery.

DynaWorld rule:

```text
If we ask generated helpers to expose components, those component names become
part of the search space. Keep them stable, semantic, and evaluator-aligned.
```

### Batch Sampling Hides Fragility

Eureka's K=16 sampling can make execution success look robust even if many
individual samples fail. The batch may contain one good reward, but failure rate
still matters for cost and reproducibility.

DynaWorld rule:

```text
Track execute_rate separately from best_score.
Do not let one lucky candidate hide a bad prompt contract.
```

## DynaWorld Mapping

### Boundary: Shaping Code Versus Fitness Code

Eureka should directly influence the local `alpha_evolve` contract:

```text
editable:
  shaped loss terms
  diagnostic summaries
  score compressors
  curriculum schedules
  prompt reflection text
  candidate-generated tests for visible behavior

not editable:
  hidden acceptance evaluator
  heldout split definitions
  data-contract checks
  leakage checks
  baseline comparison code
  final promotion thresholds
```

This is the biggest paper-006 design rule. Generated reward code is powerful
because it is white-box and inspectable; it is dangerous if it becomes the
judge.

### Microlib: `reward_reflection`

Build a local microlib that mirrors Eureka's reflection mechanism before
attempting reward/loss evolution.

Proposed contract:

```text
input:
  evaluator JSON for one candidate
  component traces over stages/checkpoints
  failure logs
  immutable problem contract

output:
  prompt-facing reflection block
  compact score signature
  list of component diagnoses
  suggested mutation class
```

Useful unit fixtures:

- all components flat;
- one component magnitude dominates;
- task score improves while hidden leak flag fails;
- candidate passes smoke but regresses heldout;
- execution failed before metric logs exist;
- user reflection overrides automated metric preference.

This microlib is low-risk and central. It can be tested without expensive
training.

### Microlib: `shaped_loss_candidate`

For training-related DynaWorld work, a Eureka-shaped candidate should edit a
small shaped loss helper, not the trainer:

```text
def compute_candidate_loss_terms(batch, outputs, context) -> tuple[total, components]:
    ...
```

The candidate must return named components. The evaluator trains or smokes with
the candidate loss, then selection uses an external metric:

```text
selection_score = F(policy_or_model_after_training)
```

For DynaWorld, F might be:

- heldout novel-view proxy score;
- no-leakage status;
- render/feature finite status;
- short-run quality curve;
- user-visible video/artifact check;
- throughput and memory budget.

The generated loss helper should not be allowed to read heldout labels unless
the contract explicitly permits them for the mode being tested.

### Microlib: `context_pruner`

Eureka's environment-as-context works because the context is pruned. A local
equivalent should produce prompt packs from code and configs:

```text
source paths
allowed symbols
forbidden symbols
visible metrics
editable function signatures
hidden evaluator summary
```

This is a good early target for `codex exec` evolution because the evaluator can
use synthetic repo snippets with known forbidden leaks and expected allowed API
exposure.

### Microlib: `component_trace_logger`

Eureka requires generated reward components to be logged. DynaWorld needs the
same for candidate helper outputs:

```text
candidate_id
stage
step
component_name
value
min
mean
max
trend
finite
```

This helps reflection and makes failures auditable. It also makes metric hacking
more visible: a candidate with high total score and nonsensical components is
easier to reject.

### Human Reflection In The Loop

DynaWorld has many cases where the user's visual or conceptual judgment matters:

- video looks blurry despite scalar improvement;
- novel view seems to leak same-view identity;
- rendered artifacts look stable but wrong;
- training appears to exploit a shortcut;
- docs/pitch wording preserves a thesis that metrics do not capture.

Eureka suggests adding a `human_reflection.md` input per microlib:

```text
Current best candidate:
Observed issue:
Desired behavior:
Forbidden shortcut:
Metric that should not be changed:
```

The runner can append this as reflection for the next mutation, but it should
not silently change the score formula. A score formula change is a new microlib
version.

### Batch Sampling Policy

Eureka uses K=16 because many rewards can fail and GPU simulation is parallel.
For local `codex exec`, the equivalent should be configurable:

```text
batch_size = 1 for expensive repo patches
batch_size = 4 for cheap pure-function microlibs
batch_size = 8+ only for synthetic/unit-only tasks
```

The candidate database should always store:

```text
execute_rate
best_score
median_score
failure_type_histogram
```

Otherwise the loop will overvalue lucky best-of-K sampling.

## Falsification Tests

### Test 1: Shaping/Fitness Separation

Create a toy microlib where candidate code can improve a visible shaping score
while a hidden fitness detects cheating.

Claim falsified if evolved candidates improve the visible metric while hidden
fitness regresses and the runner still selects them.

### Test 2: Reflection Ablation

Run the same microlib with:

```text
scalar_score_only
scalar_plus_component_traces
scalar_plus_component_traces_plus_error_summary
```

Claim falsified if component reflection does not improve repair rate, candidate
quality, or failure-type diversity over scalar-only feedback.

### Test 3: Context Leakage Audit

Give the prompt packer a synthetic source tree with allowed train variables and
forbidden heldout variables.

Claim falsified if the prompt exposes forbidden names, paths, or evaluator code.

### Test 4: Component Trace Sanity

Generate candidate helpers that return named components. Feed them through the
logger and reflection builder.

Claim falsified if flat, dominant, NaN, or disconnected components do not
produce distinct reflection diagnoses.

### Test 5: Batch Sampling Versus Evolution

Compare:

```text
one iteration with K=16 independent candidates
four iterations with K=4 and reflection
```

Claim falsified if iterative reflection does not beat or at least change the
search distribution relative to wider first-shot sampling.

### Test 6: Human Reflection Safety

Add a human reflection asking for a qualitative behavior improvement while
leaving the hidden metric unchanged.

Claim falsified if the runner mutates the score formula or hidden gate instead
of only changing the candidate artifact/prompt reflection.

## Notes For Future Papers

- Voyager should be compared on whether its skill library gives a safer memory
  substrate than Eureka's Markovian last-best reflection.
- Reflexion and Self-Refine should be read through the reward-reflection lens:
  do verbal memories actually provide component-level credit assignment?
- SWE-agent and Agentless should challenge whether evolving reward/loss helpers
  is too expensive for repo work and whether simpler bounded repair suffices.
- CodeT matters because generated tests resemble generated reward code: useful
  as visible shaping, dangerous as final judge.

## Bottom Line

Eureka adds the metric-hacking boundary that the local AlphaEvolve plan needed:

```text
Codex may evolve shaping code.
Codex may evolve reflection summaries.
Codex may evolve visible helper tests.
Codex must not evolve the hidden final evaluator.
```

For DynaWorld, the first Eureka-shaped work should not be full training reward
evolution. It should be the cheaper infrastructure that makes reward evolution
auditable: context pruning, component traces, reflection generation, and
immutable fitness separation.
