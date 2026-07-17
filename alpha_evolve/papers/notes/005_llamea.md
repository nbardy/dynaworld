# 005 - LLaMEA: A Large Language Model Evolutionary Algorithm for Automatically Generating Metaheuristics

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2405.20132
    https://arxiv.org/pdf/2405.20132
    https://github.com/XAI-liacs/LLaMEA
    https://xai-liacs.github.io/LLaMEA/
    https://www.nikivanstein.nl/projects/llamea

Bibliographic metadata:
    Authors: Niki van Stein, Thomas Back.
    First arXiv submission: 2024-05-30.
    Latest arXiv version inspected: v4, 2025-01-30.
    Venue status on arXiv page: accepted at IEEE TEVC.

Why this paper matters for alpha_evolve:
    LLaMEA is the cleanest small evolutionary loop in this queue so far. It is
    not a large multi-agent platform. It is one executable candidate, one
    evaluator, one score, one feedback string, and a parent-selection rule. That
    makes it a practical first runner for DynaWorld before building the full
    CodeEvolve/AlphaEvolve program database. The maintained implementation also
    exposes exactly the later knobs we probably want: diff mode, HPO, niching,
    population evaluation, evaluation timeouts, parallel workers, and structured
    solution metadata.

One-sentence mechanism:
    Use an LLM to generate a Python algorithm, execute it under a hard benchmark
    budget, feed score/error feedback back into the prompt, and select either
    the best-so-far parent or the latest parent for the next mutation.

## Reading Questions

- What is the executable feedback signal?
  AOCC, the area over a convergence curve, measured on BBOB functions through
  IOHexperimenter. Errors and failed executions are also feedback, and failed
  candidates receive the worst score.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Executable Python classes implementing black-box optimization algorithms. The
  search object is code, but the loop also searches the LLM's implicit algorithm
  design space because each mutation prompt can refine or redesign the parent.

- What is the population/database/selection mechanism?
  The core paper uses a minimal evolutionary strategy: initial candidate, then
  repeated offspring generation. It compares an elitist `(1+1)` strategy, which
  mutates the best-so-far candidate, with a non-elitist `(1,1)` strategy, which
  mutates the latest candidate. The prompt includes a compact history of prior
  algorithm names and scores, but not the full code of every prior candidate.

- What evidence proves the loop improves over one-shot generation?
  The loop beats a random-search LLM baseline that repeatedly samples from the
  starting prompt, and the best `(1+1)` GPT-4 run outperforms Evolution of
  Heuristics and random sampling on the inspected BBOB setting. The result is
  still domain-bound: many discovered algorithms look like DE/CMA hybrids, and
  higher-dimensional generalization is mixed.

- What does the method assume that DynaWorld does not have?
  It assumes a compact benchmark with cheap, repeatable evaluations and one
  scalar objective that meaningfully summarizes progress. DynaWorld has slower
  training loops, leakage-sensitive data contracts, mixed smoke/quality gates,
  and objectives where scalar aggregation can hide bad behavior.

## Mechanism

LLaMEA's base problem is black-box optimizer design. The LLM is not solving
BBOB directly. It is writing an optimizer that will later be run on many BBOB
functions under a strict function-evaluation budget.

The generated code interface is deliberately narrow:

- The candidate is a Python algorithm class.
- It has an initializer with a budget and dimension.
- It has a callable method that receives a black-box function.
- It may call the objective only within the allowed evaluation budget.
- It must respect a bounded search space.
- It is asked to provide a name and a short description.

This is the important transferable design: the task prompt defines a rigid code
contract, not a vague research request. The LLM's creativity is inside that
contract.

The initial prompt contains:

- a role/task description;
- the benchmark constraints;
- the exact Python interface;
- allowed dependencies;
- a small example algorithm;
- output formatting requirements.

The example algorithm matters. It gives the model a runnable reference and
reduces extraction failure. It also biases the search. For DynaWorld, seed
examples should be intentionally simple and contract-faithful, not clever
wrong abstractions that the loop will copy forever.

The paper's loop can be understood as:

```text
task_prompt
parent = LLM(task_prompt)
score, std, error = evaluate(parent)
best = parent

for generation in budget:
    selected = best        # (1+1), elitist
    selected = latest      # (1,1), non-elitist alternative

    mutation_prompt =
        task_prompt
        previous_names_and_scores
        selected_full_code
        selected_score_std_and_error
        short instruction to refine or redesign

    child = LLM(mutation_prompt)
    child_score, child_std, child_error = evaluate(child)
    latest = child

    if child_score improves best:
        best = child
```

The most useful phrase in the mutation instruction is the freedom to either
refine or redesign. That lets the LLM choose between local mutation and restart
without needing a separate explicit operator. This is weaker than EoH's named
operator taxonomy, but much cheaper to implement.

The `(1+1)` and `(1,1)` distinction is central:

- `(1+1)` is safe exploitation. It keeps the best code visible and tends to
  avoid losing a good design.
- `(1,1)` is more exploratory. It can walk away from a local optimum because the
  next parent is the latest candidate, not the best candidate.

For the local runner this suggests three selector modes:

```text
best_so_far      # LLaMEA (1+1)
latest_parent    # LLaMEA (1,1)
mixed            # mostly best_so_far, occasional latest_parent
```

LLaMEA uses a compact history trick that is more important than it looks. It
does not stuff every previous program into the context. It includes prior names
and scores, then only the selected candidate's full code. This gives the LLM
enough memory to avoid obvious duplicates without spending context on
irrelevant old code.

For DynaWorld, the history line should probably be:

```text
candidate_id | short_name | operator | score | stage_pass_depth | score_signature | brief_failure
```

Full code should be included only for the selected parent and maybe one
contrasting inspiration. That matches FunSearch/CodeEvolve without paying for a
full archive prompt every generation.

### Error Handling

LLaMEA treats errors as first-class evaluation results. Extraction, syntax,
import, and runtime failures are caught, recorded, and fed back. Failed
candidates receive a zero score.

That is a stronger signal than a generic "the code failed" note. The maintained
implementation's `Solution` object stores:

- code;
- name;
- description;
- config space for HPO;
- generation;
- fitness;
- feedback;
- error;
- parent IDs;
- operator;
- metadata;
- task prompt.

The exact object does not need to be copied, but the fields are a good local
schema. For `alpha_evolve`, an error should include:

```text
stage_failed
command
return_code
timeout
exception_type
exception_summary
stderr_tail
log_path
score_for_selection
```

Errors should not be hidden outside the candidate database. A failed candidate
is often the next prompt's best training example.

### Prompt Feedback Shape

The paper compares aggregate feedback against more detailed per-function-group
feedback. The detailed feedback does not improve the reported results. That is
a useful warning: more metric text is not automatically better prompt signal.

For DynaWorld, this does not mean "only use one scalar." It means detailed
feedback should be compressed into a score signature and short diagnosis rather
than dumping full logs. A good local feedback block is:

```text
Overall score: 0.731
Passed stages: syntax, unit, smoke_f3
Failed stage: smoke_f32
Score signature: finite=yes, leak=no, grad=yes, speed=medium, loc_delta=small
Best regression: f32_pca_video_missing
Relevant stderr: AttributeError on FeatureToColor path...
Instruction: repair the failure without changing forbidden files.
```

This is close to LLaMEA's selected code plus score/std/error prompt, but adapted
to codebase work where "why it failed" matters more than one numeric std.

### Current Maintained Library Surface

The paper is the algorithm. The current `XAI-liacs/LLaMEA` repository is a more
general framework. The README and code list several features worth tracking for
our own design:

- `n_parents`, `n_offspring`: population sizes.
- `budget`: generation budget.
- `niching`: diversity modes, including sharing, clearing, novelty, and
  map-elites in the current code.
- `evaluate_population`: batch evaluation mode for expensive candidate
  evaluation.
- `diff_mode`: ask the LLM for unified diff patches instead of full code.
- `HPO`: in-the-loop hyperparameter optimization.
- `eval_timeout`: evaluation timeout to prevent infinite loops.
- `max_workers` and `parallel_backend`: parallel evaluation controls.
- `adaptive_mutation` and `adaptive_prompt`: co-adapt mutation strength and
  prompt text.

This is a roadmap, not a starting scope. The first DynaWorld version should
copy the paper's minimal loop, then add these features only when the evaluator
is already trusted.

## Evaluation

LLaMEA evaluates generated optimizers on the BBOB benchmark through
IOHexperimenter/IOHprofiler. BBOB contains 24 noiseless functions grouped into
families such as separable, ill-conditioned, unimodal, and multimodal problems.
Each candidate optimizer is run with a fixed function-evaluation budget.

The primary metric is AOCC, area over the convergence curve. The important
property is that it is anytime-aware: an algorithm that reaches good solutions
early can score well even if final performance ties later. This fits optimizer
design better than only scoring the terminal value.

For local `alpha_evolve`, AOCC maps to evaluator curves such as:

- loss improvement area over the first N train steps;
- quality proxy area over a fixed smoke budget;
- compile/test pass depth over time;
- render throughput trajectory under a fixed scene/frame budget;
- number of useful artifacts produced before timeout;
- regression-free improvement over a baseline curve.

The paper trains/generates on 5D BBOB and then checks 10D and 20D behavior. The
best algorithms remain competitive in some regimes, but CMA-ES can dominate at
longer budgets/higher dimensions. The honest lesson is not "LLM-evolved code
generalizes." It is "generalization must be explicitly tested after discovery."

For DynaWorld, every evolved microlib needs a train/eval split:

```text
development evaluator     # cheap, used for selection
heldout evaluator         # less frequent, prevents overfitting to smoke
export/acceptance gate    # not visible in full detail to the prompt
```

The paper compares against:

- CMA-ES;
- Differential Evolution;
- Evolution of Heuristics;
- random LLM sampling from the starting prompt.

The random LLM baseline is especially important. We should not claim local
evolution works until it beats:

```text
N independent codex exec samples with the same evaluator budget
```

If the evolutionary loop only matches independent sampling, the database and
selection machinery are not buying anything yet.

### What The Results Actually Prove

The paper proves that a simple generate/evaluate/mutate loop can produce useful
optimizer code under a strong benchmark harness. It also shows that the LLM can
incorporate runtime failures and metric feedback across generations.

It does not prove:

- open-ended scientific novelty;
- robustness outside the benchmark family;
- correctness under messy repo constraints;
- immunity to benchmark overfitting;
- that detailed metric dumps improve search;
- that the best algorithms are more than clever recombinations of known
  optimizer families.

That limited claim is still enough for DynaWorld. We only need a loop that beats
one-shot Codex on tightly-scored local problems before building a broader
agentic system.

## Failure Modes

### Prompt Bias Toward Known Families

Many generated optimizers resemble Differential Evolution, CMA-ES, or hybrids.
That may be because these families are strong on BBOB, because they are common
in training data, or both. Either way, the LLM's prior strongly shapes the
search.

DynaWorld implication: if the seed prompt names the wrong local abstraction, the
loop may spend generations decorating it. Prompts should name the contract and
the failure, not over-prescribe the implementation family unless we really want
that family.

### Scalar Metric Erases Behavior

AOCC is useful, but a single aggregated score can hide per-family failures. The
paper's detailed feedback did not improve search, but that does not make the
hidden failures harmless.

DynaWorld implication: store vector metrics and score signatures even if the
prompt receives a compressed summary. Selection can use a scalar, but audit and
archive placement need richer descriptors.

### High-Dimensional Generalization Is Not Free

Generating on 5D and testing on 10D/20D is a good experiment. The mixed result
is a warning. Search can produce code that is tuned to the cheap discovery
environment.

DynaWorld implication: cheap smoke should not be the only gate. A candidate that
wins one-step smoke can still fail real frame counts, feature dimensions,
heldout camera paths, or longer optimizer steps.

### Detailed Feedback Can Add Noise

The detailed per-function-group feedback did not improve results. This is
counterintuitive if we assume more observation always helps.

DynaWorld implication: do not paste entire logs into `codex exec`. Summarize. Use
stable short fields. Keep the full logs in artifacts, and include a path for the
agent to inspect only when needed.

### Runtime And Extraction Are Part Of The Objective

Generated code can fail to parse, import, execute, or finish. LLaMEA handles
that with error feedback and zero score.

DynaWorld implication: the evaluator must grade operational behavior:

- does the patch apply?
- do allowed paths stay within bounds?
- does the command finish before timeout?
- does it write unexpected files?
- does it pass syntax/import gates?
- does it preserve config/data-contract invariants?

This cannot be an afterthought because Codex patch evolution will otherwise
optimize for code that looks plausible but is not runnable.

### HPO Confounds Structural Claims

The paper's supplemental HPO discussion shows that numeric tuning can help,
especially in higher dimensions. But HPO can blur whether the LLM discovered a
better algorithmic structure or merely exposed knobs that another optimizer
tuned.

DynaWorld implication: separate structural evolution from constants/config
tuning. First evolve a better helper or patch family. Then run mechanical HPO or
grid/random search over its exposed constants.

### Evaluation Cost Can Dominate Search Quality

LLaMEA's loop is feasible because BBOB runs are cheap enough to repeat many
times. DynaWorld evaluations can be expensive.

DynaWorld implication: early microlibs should target cheap correctness and
throughput problems, not full training quality. The right first targets are
where a candidate can be judged in seconds or minutes.

## DynaWorld Mapping

### Minimal Runner: `llamea_serial`

The first `alpha_evolve` implementation should have a LLaMEA-style serial
runner before islands, migration, MAP-Elites, or prompt evolution.

Proposed directory shape:

```text
alpha_evolve/
  runners/
    llamea_serial.py
  core/
    candidate.py
    candidate_store.py
    codex_adapter.py
    evaluator.py
    parent_selector.py
    prompt_pack.py
    score_signature.py
    patch_guard.py
  microlibs/
    <problem_name>/
      contract.md
      seed.py
      evaluator.py
      problem.jsonc
      prompts/
        initial.md
        mutate.md
      candidates/
      logs/
```

The runner should be boring:

```text
load microlib contract
seed candidate or request initial candidate from codex exec
evaluate candidate
repeat generation budget:
    select parent with best_so_far/latest/mixed
    build mutation prompt
    call codex exec with prompt text
    parse patch or file artifact
    apply into candidate worktree
    run evaluator cascade
    record score, signature, logs, errors
    update best
```

This is intentionally smaller than CodeEvolve. It gives us a baseline loop to
measure before adding islands.

### Codex Exec Adapter

The Codex call should be isolated behind `codex_adapter.py`. The initial goal
used the shorthand `codex -p "..."`, but the checked local CLI uses `-p` for
`--profile`; prompt text is positional, and non-interactive runs should use
`codex exec "<prompt>"`. The adapter should hide that detail so a future CLI
change does not leak through the runner.

The adapter should own:

- prompt file creation;
- command construction for `codex exec`;
- working directory selection;
- environment variables;
- timeout;
- stdout/stderr capture;
- response artifact path;
- extraction of patch or code block;
- normalization into a candidate artifact.

The rest of the runner should not know shell syntax. It should call:

```python
candidate_artifact = codex.generate(prompt, worktree, mode="patch")
```

For early runs, do not let Codex edit the real repo directly. Generate in a
candidate worktree or isolated copy, then evaluate there.

### Candidate Store

LLaMEA's `Solution` object maps almost directly to a local JSONL row:

```json
{
  "candidate_id": "uuid",
  "generation": 7,
  "parent_ids": ["..."],
  "selector": "best_so_far",
  "operator": "refine_or_redesign",
  "name": "short descriptive name",
  "thought": "why this change should help",
  "artifact_kind": "patch",
  "artifact_path": "candidates/0007.patch",
  "worktree_path": "candidates/0007/worktree",
  "score": 0.731,
  "score_signature": {
    "finite": true,
    "leak_free": true,
    "passed_stage": "smoke_f3",
    "failed_stage": "smoke_f32",
    "speed_bucket": "medium",
    "loc_delta_bucket": "small"
  },
  "feedback": "short prompt-facing feedback",
  "error": "short error summary",
  "log_path": "logs/0007.json",
  "created_by": "codex exec",
  "task_prompt_hash": "..."
}
```

This schema keeps paper-005 minimalism while preserving the CodeEvolve fields we
will need later.

### Prompt Template

The mutation prompt should follow LLaMEA's compact shape:

```text
SYSTEM/ROLE
You are improving a bounded DynaWorld microlib. Obey the contract exactly.

IMMUTABLE CONTRACT
- allowed files
- forbidden files
- evaluator commands
- score definition
- output format
- timeout and dependency constraints

ARCHIVE SUMMARY
candidate_id | name | operator | score | stage | signature | failure
...

SELECTED PARENT
score, signature, feedback, error
patch/code

TASK
Either refine or redesign the parent to improve the evaluator score.
Keep the patch small. Do not change the evaluator. Return only the requested
artifact format.
```

This retains LLaMEA's "selected parent plus history list" without importing all
of AlphaEvolve's complexity.

### AOCC-Style Local Scores

The most direct transferable metric is anytime scoring. DynaWorld should avoid
only final-step metrics when early behavior matters.

Candidate local metrics:

- train-loss area over first 1, 2, 4, 8, 16 steps;
- render time area over repeated frames;
- validation artifact completeness over step budget;
- memory use or failure count over a fixed command budget;
- staged gate area, where deeper gates contribute more only after earlier gates
  pass.

A staged AOCC-like score could be:

```text
score =
    syntax_weight * syntax_pass
  + unit_weight * unit_pass
  + smoke_weight * smoke_curve_area
  + quality_weight * normalized_quality_curve
  - time_penalty
  - loc_penalty
  - regression_penalty
```

The exact formula belongs in each microlib. The shared rule is that the prompt
sees the scalar plus a compact signature; the database stores the full vector.

### First DynaWorld Problem Targets

Good early targets are cheap, bounded, and hard to hand-wave.

1. Config normalization cleanup

   Evolve small refactors that remove repeated config `.get` and alias
   anti-patterns while preserving the 1-step smoke. This is code-health oriented
   and easy to evaluate with detectors plus smoke, but it risks optimizing for
   superficial LOC reduction.

2. Score-signature helper

   Evolve the helper that maps evaluator JSON into archive descriptors and
   prompt-facing summaries. This bootstraps the rest of `alpha_evolve` and has a
   strong unit-testable contract.

3. Prompt-pack compressor

   Evolve the function that chooses which candidate history lines and failures
   enter the next prompt. The evaluator can use synthetic candidate archives to
   test token budget, diversity, parent visibility, and determinism.

4. Renderer microbenchmark selector

   Evolve a bounded policy that selects renderer variant or batching strategy
   under known constraints. The evaluator can run a small benchmark matrix and
   reject any candidate that changes renderer semantics.

5. Sampler/leakage audit helper

   Evolve helper code that validates same-view versus heldout-view data
   contracts. This is valuable because the project state says mixed same-view
   plus novel-view sampling is a next bridge, but the evaluator must be strict
   enough to prevent leakage.

6. Feature-video/logging robustness

   Evolve helpers around feature colorization/PCA/video logging paths. This is
   useful if failures are frequent and cheap to reproduce, but it should not
   become a broad trainer rewrite target.

Avoid as first targets:

- whole trainer rewrites;
- renderer kernel rewrites without a fast parity gate;
- full training quality optimization;
- changes requiring W&B/network state;
- tasks where the evaluator is mostly subjective.

### How This Changes The Local Plan

Before LLaMEA, the synthesis pointed toward CodeEvolve-style islands as the
first real architecture. After LLaMEA, the better sequence is:

```text
serial LLaMEA-style runner
beat one-shot codex exec on one microlib
add operator labels from EoH
add bounded patch mode from CodeEvolve
add islands/archive diversity only after serial loop has signal
```

The serial loop is also a debugging tool. If the evaluator, candidate store, and
Codex adapter cannot support `(1+1)` evolution, islands will only multiply
failure modes.

## Falsification Tests

### Test 1: Serial Evolution Versus Random Codex Sampling

Pick one small microlib. Run:

```text
10 independent codex exec samples from the initial prompt
10 generations of llamea_serial with the same total Codex-call budget
```

Claim falsified if serial evolution does not beat independent sampling on best
score, median score, and deepest passed evaluator stage.

### Test 2: Best-So-Far Versus Latest Parent

Run the same microlib with:

```text
selector = best_so_far
selector = latest_parent
selector = mixed
```

Claim falsified if there is no measurable difference in diversity, recovery
from failed candidates, or final score. In that case selector complexity is not
worth adding yet.

### Test 3: Error Feedback Ablation

Run with and without structured error feedback in the prompt.

Claim falsified if structured error feedback does not reduce repeated failure
types or improve recovery after the first failed candidate.

### Test 4: Aggregate Versus Detailed Feedback

Compare prompt feedback modes:

```text
scalar_only
scalar_plus_signature
full_log_summary
```

Expected result from LLaMEA: scalar plus compact signature should beat raw full
log dumps. If full logs win, our compressor is losing useful information.

### Test 5: AOCC Versus Final Score

Run a microlib where early behavior and final behavior can diverge. Compare
selection by final score against selection by curve area.

Claim falsified if curve scoring selects candidates that look better early but
consistently fail heldout acceptance.

### Test 6: HPO Separation

Take a structural candidate family found by code evolution. Freeze the code and
run a small mechanical constant search.

Claim falsified if most gains come from constants available in the seed code,
not structural evolution. In that case the microlib should be a config/HPO
problem, not a Codex code-evolution problem.

### Test 7: Prompt Bias Check

Seed two initial prompts with different examples for the same evaluator. Measure
whether the discovered candidates collapse into the example's family.

Claim falsified if the runner cannot escape the example style. The prompt then
needs weaker examples or explicit alternative operators.

## Notes For Future Papers

- Eureka should clarify how to evolve reward/evaluator code without letting the
  model corrupt the metric.
- Voyager should be read for skill-library persistence: LLaMEA has candidate
  history, but not a reusable skill library.
- Reflexion/Self-Refine should be compared against LLaMEA error feedback:
  natural-language failure memories may help, but only if they stay grounded in
  evaluator artifacts.
- SWE-agent and Agentless should pressure-test whether `codex exec` needs a rich
  agent loop or whether simple localization plus bounded repair is enough.
- CodeT matters for evaluator bootstrapping, but generated tests must be kept
  separate from hidden acceptance gates.

## Bottom Line

LLaMEA is the first implementation baseline to build, not the final system. It
gives DynaWorld a small falsifiable loop:

```text
one bounded microlib
one selected parent
one compact history list
codex exec mutation
hard evaluator cascade
structured error feedback
best-so-far or latest-parent selection
```

If that loop cannot beat independent `codex exec` sampling, the project should not
spend complexity on islands, MAP-Elites, prompt evolution, or multi-agent
coordination yet.
