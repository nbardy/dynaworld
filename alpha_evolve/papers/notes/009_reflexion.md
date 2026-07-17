# 009 - Reflexion: Language Agents with Verbal Reinforcement Learning

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2303.11366
    https://arxiv.org/pdf/2303.11366
    https://github.com/noahshinn/reflexion

Implementation artifacts inspected:
    https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/generate_reflections.py
    https://github.com/noahshinn/reflexion/blob/main/alfworld_runs/alfworld_trial.py
    https://github.com/noahshinn/reflexion/blob/main/hotpotqa_runs/agents.py
    https://github.com/noahshinn/reflexion/blob/main/hotpotqa_runs/react.py
    https://github.com/noahshinn/reflexion/blob/main/programming_runs/reflexion.py
    https://github.com/noahshinn/reflexion/blob/main/programming_runs/reflexion_ucs.py
    https://github.com/noahshinn/reflexion/blob/main/programming_runs/generators/py_generate.py
    https://github.com/noahshinn/reflexion/blob/main/programming_runs/executors/py_executor.py

Bibliographic metadata:
    Authors: Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath,
    Karthik Narasimhan, Shunyu Yao.
    First arXiv submission: 2023-03-20.
    Latest arXiv version inspected: v4, 2023-10-10.
    Venue/context: NeurIPS 2023.

Why this paper matters for alpha_evolve:
    Reflexion is the missing memory layer between ReAct repair and evolutionary
    program search. It says that a failed trial should not only be archived as a
    raw trace. The failure should be compressed into a short verbal lesson that
    conditions the next attempt. For `alpha_evolve`, that maps directly to
    candidate failure notes, prompt-side lessons, and repair-loop state.

    The important boundary is that Reflexion memory is not a judge. It is a
    hypothesis about why a candidate failed. The evaluator still supplies the
    pass/fail or score. In a DynaWorld Codex evolver, reflective memory can make
    `codex exec` less repetitive, but only hard evaluator artifacts can promote
    a patch, microlib, or skill.

One-sentence mechanism:
    After each failed trial, use an LLM self-reflection pass to convert sparse
    evaluator feedback plus the trajectory into a concise natural-language
    lesson, store a bounded number of those lessons, and include them in the
    next actor attempt.

## Reading Questions

- What is the executable feedback signal?
  It depends on the task. ALFWorld uses environment success, loop/hallucination
  heuristics, or an LLM evaluator. HotPotQA uses exact-match answer grading.
  Programming uses generated unit tests as visible feedback and heldout tests
  for final evaluation. The paper's key claim is not that one feedback source is
  always correct, but that sparse feedback becomes more useful when translated
  into a compact verbal correction.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Reflexion searches over repeated attempts by one prompt-conditioned agent. In
  programming, it searches over code implementations and their repairs. In
  reasoning and ALFWorld, it searches over trajectories. The policy parameters
  are the LLM plus the memory buffer, not model weights.

- What is the population/database/selection mechanism?
  There is no evolutionary population in the core method. Selection is
  per-problem and sequential: keep trying until the evaluator passes or the
  trial budget ends. Memory is a small sliding window, usually one to three
  reflections. The programming repo also includes a UCS variant that branches
  over test-state signatures, but the main idea is still reflection-guided
  retry, not population evolution.

- What evidence proves the loop improves over one-shot generation?
  The paper reports improvements on ALFWorld, HotPotQA, HumanEval, HumanEval
  Rust, MBPP Rust, and LeetcodeHard. It also includes ablations showing that
  generated tests without reflection, or reflection without execution feedback,
  can underperform. For alpha_evolve, the ablations matter more than the headline
  results: blind retry is weak, raw trajectory memory is weaker than distilled
  reflection, and false-positive internal tests can hurt.

- What does the method assume that DynaWorld does not have?
  It often assumes tasks repeat in a way that makes a reflection useful for the
  exact same problem. DynaWorld evolution may mutate code across related but not
  identical benchmark targets. Reflections must therefore be keyed by problem,
  evaluator version, candidate family, and visible metric signature, otherwise
  stale lessons can poison later generations.

## Mechanism

Reflexion introduces three roles:

```text
Actor Ma:
    generates an answer, action trajectory, or code implementation

Evaluator Me:
    scores the attempt using environment reward, exact match, tests, heuristics,
    or an LLM evaluator

Self-reflection model Msr:
    converts the trajectory plus evaluator feedback into a verbal lesson
```

The policy is:

```text
pi_theta(action | state)
theta = {actor LLM, memory}
```

This is a useful framing because the learning is in context. The model weights
do not change. What changes is the memory passed to the actor.

The loop:

```text
attempt 0:
    actor produces trajectory/code
    evaluator returns score or pass/fail
    reflection model writes sr_0
    memory = [sr_0]

attempt t:
    actor sees task + bounded reflection memory
    actor produces new trajectory/code
    evaluator returns score or pass/fail
    reflection model writes sr_t
    append sr_t to memory
    trim memory to capacity
```

The paper distinguishes:

```text
short-term memory:
    the current trajectory or current attempt details

long-term memory:
    distilled self-reflections from prior attempts
```

That distinction is the part to copy locally. The raw stdout, diff, command
logs, and patch all belong in the candidate archive. The next prompt should not
receive all of that by default. It should receive a small reflection that says
what the previous attempt misunderstood.

### Actor

The actor can be CoT, ReAct, or a code-generation prompt. Reflexion is not tied
to one action protocol. In HotPotQA it can sit on CoT or ReAct. In programming
it sits on a code writer. In ALFWorld it sits on a ReAct-style action loop.

For `alpha_evolve`, this means reflection should be a layer around existing
candidate generators:

```text
codex exec one-shot patch
codex exec ReAct repair loop
codex exec LLaMEA-style mutate parent
codex exec island migration mutation
```

The reflection memory should not force all generators into one architecture.
It should be a prompt-side optional context object.

### Evaluator

Reflexion's evaluator is intentionally modular:

```text
reasoning:
    exact match answer grading

decision-making:
    environment success, loop heuristics, or LLM evaluator

programming:
    generated visible unit tests for feedback and hidden benchmark tests for
    final pass/fail
```

The programming setup is the closest to DynaWorld. It uses generated unit tests
as an internal feedback source. If those tests pass, it checks the real benchmark
test. That is effectively a two-stage evaluator:

```text
visible generated tests:
    shape reflection and allow early repair

heldout real tests:
    decide whether the item is solved
```

Local analogy:

```text
candidate-visible stage:
    syntax, unit tests, one-step smoke, cheap benchmark slice, diagnostic score

hidden/repo-owned stage:
    heldout manifest, rerun verifier, baseline table update, manual promotion
```

Codex-generated tests can be a useful shaping signal, but they must not be the
promotion gate.

### Self-Reflection

The self-reflection pass receives:

```text
task
previous implementation or trajectory
visible evaluator feedback
maybe prior reflections
```

It emits:

```text
a few sentences explaining the mistake and how to try again
```

The released programming prompt is direct: explain why the implementation is
wrong as indicated by unit tests, and provide only the short description, not a
new implementation. The ALFWorld prompt similarly asks for a concise plan that
accounts for the mistake with specific actions that should have been taken.

This separation is important:

```text
reflection pass:
    diagnose and compress

actor pass:
    generate the next patch/code/trajectory
```

Do not ask one `codex exec` call to both fully diagnose and produce a large
mutation if the failure is messy. A cheap reflection step can make the next
mutation prompt smaller and more focused.

## Programming Implementation

The released `programming_runs/reflexion.py` loop is directly relevant:

```text
for each dataset item:
    generate internal tests
    generate first function implementation
    execute implementation on internal tests
    if internal tests pass:
        evaluate on real benchmark tests
        exit if solved
    while under max_iters:
        generate self-reflection from implementation + test feedback
        generate improved implementation from previous impl + feedback + reflection
        execute internal tests
        if internal tests pass or final iteration:
            evaluate on real benchmark tests
            exit if solved
```

The executor returns structured feedback:

```text
Tests passed:
    ...

Tests failed:
    assert ... # output: ...
```

That shape should inform local evaluator output. A future DynaWorld evaluator
should not return one scalar unless the problem is truly scalar. It should
return a compact, model-readable failure object:

```text
gate: train_import
status: failed
command: ...
stderr_tail: ...
top_metric_regression: ...
first_bad_artifact: ...
likely_contract: config schema / renderer dispatch / data contract / baseline
```

The paper's generated-test hazard is also visible in the implementation. If the
visible tests pass, the loop checks hidden benchmark tests. If the visible tests
are false positives, the method can prematurely accept a bad candidate. This is
exactly the danger for generated DynaWorld tests: they help repair but cannot
certify.

### Reflexion-UCS

The repo includes `reflexion_ucs.py`, which branches code attempts and uses a
test-state signature as the state id:

```text
state = tuple(test_i_passed for test_i in internal_tests)
cost = number_of_failing_tests
goal = all tests pass
```

The expansion step:

```text
generate multiple improved implementations
execute each on internal tests
reflect on each failure
search toward fewer failing tests
```

This is not the main paper method, but it is useful for `alpha_evolve` because
it is a small bridge from serial reflection to search. For our work, the analog
would be:

```text
state signature:
    tuple(stage_passed, metric_bin, regression_flags, artifact_flags)

cost:
    weighted failure count or negative visible score delta

expand:
    codex exec mutation conditioned on reflection and parent patch
```

The risk is obvious: if the state signature only sees shallow tests, search will
overfit shallow tests. Use it first for microlibs with reliable deterministic
gates.

## Experiments And Results

### ALFWorld

Setup:

- 134 ALFWorld text environments.
- Six task families such as finding hidden objects and moving/manipulating
  objects.
- ReAct is the actor.
- A self-evaluation trigger fires when the agent repeats the same action and
  observation more than three cycles, or exceeds 30 actions.
- Memory is truncated to the last three reflections.

Result:

- ReAct + Reflexion solves 130 of 134 tasks in the reported heuristic setting.
- Baseline ReAct improves for a while but stalls earlier across trials.

Mechanism lesson:

    Long trajectories hide the first important mistake. Reflection compresses a
    failed trajectory into the early wrong assumption or missed action.

Local mapping:

    A DynaWorld candidate can fail 10 minutes later because of a first wrong
    assumption about data shape, renderer dispatch, or config normalization.
    The reflection pass should point at the earliest contract violation, not
    just the final stderr line.

### HotPotQA

Setup:

- 100 HotPotQA examples.
- CoT and ReAct actor variants.
- Exact-match grading supplies binary feedback between trials.
- Memory is capped to three experiences.
- An episodic-memory ablation includes raw recent trajectory instead of
  distilled reflection.

Result:

- Reflexion improves both search/retrieval and reasoning settings.
- The paper reports that self-reflection gives an 8 percent absolute boost over
  the raw episodic-memory advantage in the CoT ground-truth-context ablation.

Mechanism lesson:

    Raw trace replay is not equivalent to reflection. The value is compression
    and interpretation, not just more context.

Local mapping:

    Do not paste entire previous logs into every Codex prompt. Store raw logs in
    the archive. Feed the next attempt a compact reflection and links/ids to the
    exact artifacts if needed.

### Programming

Setup:

- Python and Rust function generation.
- HumanEval, MBPP, translated Rust subsets via MultiPL-E, and LeetcodeHardGym.
- Generated unit tests are used for visible feedback.
- The programming loop caps memory at one reflection.

Headline results:

- HumanEval Python: Reflexion reports 91.0 pass@1 versus GPT-4 baseline around
  80 in the table.
- HumanEval Rust: 68.0 pass@1 versus GPT-4 baseline 60.0.
- MBPP Rust: 75.4 pass@1 versus GPT-4 baseline 70.9.
- Leetcode Hard Python: 15.0 versus GPT-4 baseline 7.5.
- MBPP Python underperforms the GPT-4 baseline in the table, 77.1 versus 80.1.

The MBPP Python miss is one of the most important parts of the paper. The paper
attributes the problem to generated-test quality. False positives are bad:

```text
generated visible tests pass
real solution is wrong
agent stops too early
```

The paper argues false negatives are less dangerous:

```text
generated visible tests fail
real solution may be correct
agent can still reflect and maybe preserve or repair
```

Local mapping:

    Candidate-visible tests should err toward false negatives, not false
    positives. A false negative costs extra repair. A false positive pollutes
    the program database and skill library.

### Rust Ablations

The HumanEval Rust ablation is a useful warning:

```text
base model:
    0.60

no internal test generation, self-reflection true:
    0.52

test generation true, no self-reflection:
    0.60

full Reflexion:
    0.68
```

Interpretation:

    Reflection without grounded execution feedback can be worse than baseline.
    Execution feedback without reflection can catch errors but may not produce
    repairs. Both are needed for harder code tasks.

Local mapping:

    A DynaWorld reflection loop needs both a real evaluator artifact and a
    natural-language failure compressor. Asking Codex to self-critique a patch
    without running the gate is likely noise.

### WebShop Negative Result

The paper reports a negative WebShop result: ReAct + Reflexion does not
significantly outperform ReAct across the tested 100 customer requests. The
authors explain that WebShop requires diverse exploration and that reflections
were not helpful enough to escape local minima.

This is crucial for alpha_evolve:

```text
Reflexion helps repeated correction.
It does not guarantee novelty.
```

If a target problem needs genuinely different families of algorithms,
reflection alone is the wrong tool. Use reflection inside each branch, but use
evolution, islands, randomization, novelty descriptors, or beam/MCTS search to
produce diversity.

## Core Design Lesson For DynaWorld

Reflexion should become a local memory microlib with strict semantics:

```text
reflection:
    concise, candidate-visible, model-generated hypothesis about failure

archive:
    complete raw attempt record, including logs, diffs, metrics, artifacts

verified skill:
    reusable code or tactic promoted by hard gates

evaluator:
    repo-owned judge that produces feedback and decides promotion
```

Do not collapse these.

Reflection is allowed to be wrong. The archive preserves evidence. The evaluator
decides. The skill library only stores promoted behavior.

## Proposed Microlibs

### `verbal_reflection_memory`

Responsibility:

```text
store bounded natural-language reflections keyed by target, evaluator version,
candidate family, parent id, and visible score signature
```

API sketch:

```python
memory.add(
    problem_id: str,
    evaluator_fingerprint: str,
    candidate_id: str,
    parent_id: str | None,
    visible_signature: dict,
    reflection: str,
    created_at: str,
) -> None

memory.select_for_prompt(
    problem_id: str,
    evaluator_fingerprint: str,
    parent_id: str | None,
    limit: int = 3,
) -> list[Reflection]
```

Invariants:

- Never select reflections from a different evaluator fingerprint by default.
- Never select reflections from hidden evaluation output.
- Keep reflections short enough to be prompt-stable.
- Preserve raw evidence by reference, not by dumping it into reflection text.

### `reflection_builder`

Responsibility:

```text
turn evaluator feedback + candidate trace into one concise correction
```

Prompt inputs:

```text
problem statement
candidate patch summary
visible evaluator result
stderr tail or metric diff
prior reflection, if any
instruction: diagnose, do not patch
```

Output schema:

```json
{
  "failure_summary": "...",
  "earliest_bad_assumption": "...",
  "next_attempt_rule": "...",
  "do_not_repeat": "...",
  "confidence": "low|medium|high"
}
```

The JSON shape is preferable to plain prose because it allows prompt filtering,
diff display, and stale-memory invalidation.

### `reflection_invalidator`

Responsibility:

```text
drop or quarantine reflections when the target, evaluator, data contract, or
parent family changes enough to make the lesson unsafe
```

Invalidation triggers:

- evaluator command changes;
- config schema changes;
- heldout manifest changes;
- target problem id changes;
- parent family diverges;
- reflection repeatedly predicts the wrong next action;
- hidden/promotion evaluator contradicts visible-stage improvement.

### `false_positive_guard`

Responsibility:

```text
prevent generated tests and cheap gates from certifying success
```

Implementation:

```text
visible gates can permit repair and ranking;
promotion requires repo-owned deterministic gates;
generated tests are tagged as candidate-visible shaping;
candidate database stores generated-test pass separately from promotion pass.
```

### `reflection_budget_controller`

Responsibility:

```text
decide when a failed candidate gets reflection versus direct archive only
```

Rules:

- Reflect on failures that are close, surprising, or repeated.
- Skip reflection on syntax/import failures that a deterministic formatter or
  smoke gate can already explain.
- Reflect once per failure family, not once per identical crash.
- Cap reflection tokens per generation.

## How To Use With `codex exec`

The local CLI check showed that current Codex uses:

```bash
codex exec "..."
```

not the user shorthand `codex -p "..."`, because `-p` is currently the profile
flag. The notes keep saying "Codex prompt" or `codex exec` to avoid encoding the
wrong invocation.

Suggested first local loop:

```text
1. Pick one microlib target.
2. Run baseline one-shot codex exec candidate generation.
3. Evaluate candidate on visible deterministic gates.
4. If failed, run reflection_builder on visible feedback.
5. Run codex exec again with:
       task
       parent patch summary
       evaluator feedback
       reflection JSON
       hard constraints
6. Repeat for max 3 repair attempts.
7. Only after that compare against LLaMEA-style parent selection.
```

Example actor prompt block:

```text
You are editing only alpha_evolve/microlibs/<target>.

Prior failed attempt:
    candidate_id: ...
    parent_id: ...
    visible gate: ...
    failed evidence: ...

Reflection:
    earliest_bad_assumption: ...
    next_attempt_rule: ...
    do_not_repeat: ...

Now produce a minimal patch. Do not change evaluator files. Do not change the
hidden gate. Keep the public API stable unless the target contract explicitly
allows an API change.
```

Example reflection prompt block:

```text
You are not writing code. Diagnose the failed attempt.

Task:
    ...

Candidate summary:
    ...

Visible evaluator result:
    ...

Prior reflection:
    ...

Return JSON with:
    failure_summary
    earliest_bad_assumption
    next_attempt_rule
    do_not_repeat
    confidence
```

## Failure Modes

### Reflection As Unverified Truth

If reflections are phrased too confidently, future Codex calls may treat them
as facts even when the evaluator evidence was ambiguous.

Mitigation:

```text
include confidence
include evidence reference ids
keep evaluator facts separate from reflection text
expire stale reflections
```

### Memory Poisoning

A bad reflection can poison all descendants in a candidate family.

Mitigation:

```text
record reflection lineage
allow parent-family quarantine
select at most 1-3 reflections
track whether a reflection improved the next visible score
```

### False-Positive Visible Tests

Generated or weak tests can say "pass" while the real target fails. Reflexion's
MBPP Python result is the warning case.

Mitigation:

```text
generated tests never promote
generated-test pass only unlocks hidden/repo-owned verification
store false-positive rate per generated-test family
prefer false negatives over false positives for candidate-visible shaping
```

### Local Minima And Low Diversity

Reflexion tends to refine an existing trajectory. It does not automatically
invent a very different search direction.

Mitigation:

```text
use reflection inside one branch
use islands or MAP-Elites for diverse families
trigger novelty search after repeated reflection failures
```

### Verbose Context Drag

Reflection can grow into a second log dump.

Mitigation:

```text
cap reflection count and token budget
store structured short fields
link raw artifacts by id
summarize repeated failures into one family-level lesson
```

### Evaluator Leak

If the reflection sees hidden promotion failures, it can overfit the hidden
gate and corrupt the experiment.

Mitigation:

```text
reflection_builder only receives candidate-visible gates
promotion failures can be stored for humans but not fed back automatically
keep visible and hidden evaluation artifacts in separate namespaces
```

## Falsification Tests For Local Implementation

### Test 1: Reflection Beats Raw Log Replay

Problem:
    Choose a small microlib with deterministic tests and common failure modes.

Compare:

```text
A: codex exec retry with full previous stderr/log tail
B: codex exec retry with structured reflection only
C: codex exec retry with no previous context
```

Support for Reflexion:
    B solves more cases or uses fewer tokens than A/C.

Failure:
    A is consistently better, meaning compression lost critical evidence.

### Test 2: Reflection Invalidation Catches Stale Lessons

Procedure:

```text
create reflection under evaluator fingerprint X
change evaluator command or expected schema to fingerprint Y
ask memory.select_for_prompt(...)
```

Expected:
    reflection is not selected unless explicitly allowed.

Failure:
    stale reflection enters prompt after evaluator/data change.

### Test 3: False-Positive Guard Blocks Promotion

Procedure:

```text
create candidate that passes generated tests but fails repo-owned hidden test
try to promote to verified_skill_library
```

Expected:
    promotion blocked, candidate remains archived, generated tests marked as
    shaping only.

Failure:
    generated-test pass is treated as promotion success.

### Test 4: Reflection Utility Is Measured

Procedure:

```text
for each reflection, record next attempt score delta and whether the same
failure recurred
```

Expected:
    memory can rank reflections by observed utility.

Failure:
    all reflections are treated equally forever.

### Test 5: Diversity Trigger Fires

Procedure:

```text
run 3 reflection-guided attempts with same failure family and no score delta
```

Expected:
    runner stops local repair and asks for a new parent/island/novelty branch.

Failure:
    runner keeps rephrasing the same advice indefinitely.

## How It Changes The AlphaEvolve Plan

Before Reflexion, the likely sequence was:

```text
one-shot codex exec
ReAct repair loop
LLaMEA serial parent mutation
islands/program database
```

After Reflexion, the sequence becomes:

```text
one-shot codex exec
ReAct repair loop with structured reflection memory
LLaMEA serial parent mutation with reflection summaries
program database with reflection utility tracking
islands with reflection invalidation across evaluator/problem boundaries
```

This does not make the system heavier if implemented as microlibs. The minimal
version is just:

```text
candidate_trace_schema
evaluator_feedback_schema
reflection_builder
verbal_reflection_memory
reflection_prompt_adapter
```

Everything else can be deferred.

## Open Questions

- Should reflections be generated by `codex exec` itself or by a cheaper model
  call wrapped outside Codex?
- Should a successful candidate produce a reflection, or only a reusable skill
  note after promotion?
- What is the smallest evaluator feedback schema that is enough for useful
  reflections without leaking hidden gates?
- Can reflection utility be tracked per problem class, such as config repair,
  renderer performance, data contract, or benchmark cleanup?
- When the reflection says "do not change X" and the next best patch must
  change X, how should the runner weaken or invalidate that lesson?
- Should island migration move reflections, or only promoted skills and score
  descriptors?

## Bottom Line

Reflexion is not an AlphaEvolve replacement. It is the repair-memory layer that
keeps Codex from repeating the same bad patch inside one problem. The local
implementation should be small, evaluator-bound, and aggressively invalidated.
Use it to improve retry quality; do not let it promote code, judge code, or
stand in for diversity.
