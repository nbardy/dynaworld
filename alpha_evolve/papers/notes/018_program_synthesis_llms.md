# 018 - Program Synthesis with Large Language Models

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2108.07732
    https://arxiv.org/pdf/2108.07732
    https://research.google/pubs/program-synthesis-with-large-language-models/
    https://github.com/google-research/google-research/tree/master/mbpp

Implementation artifacts inspected:
    https://github.com/google-research/google-research/blob/master/mbpp/README.md
    https://github.com/google-research/google-research/blob/master/mbpp/mbpp.jsonl

Bibliographic metadata:
    Authors: Jacob Austin, Augustus Odena, Maxwell Nye, Maarten Bosma,
    Henryk Michalewski, David Dohan, Ellen Jiang, Carrie Cai, Michael Terry,
    Quoc Le, Charles Sutton.
    First arXiv submission: 2021-08-17.

Why this paper matters for alpha_evolve:
    This paper complements HumanEval by giving a broader task-design view:
    MBPP-style problems are short natural-language-to-Python tasks with explicit
    assert tests; MathQA-Python tests code synthesis from harder text; and the
    paper studies prompt examples, sample count, fine-tuning, challenge tests,
    human feedback, and execution prediction.

    For `alpha_evolve`, the most useful transfer is not model scaling. It is how
    to author the first local microlib task suite:

```text
clear natural-language task
self-contained function/module
three visible tests
hidden challenge tests
many sampled candidates
functional correctness score
prompt-example ablations
```

One-sentence mechanism:
    Evaluate large language models from 244M to 137B parameters on MBPP and
    MathQA-Python by sampling short Python programs from natural-language
    descriptions, executing them against tests for functional correctness, and
    studying scaling, fine-tuning, prompt sensitivity, dialog feedback, and
    execution understanding.

## Reading Questions

- What is the executable feedback signal?
  Functional correctness under tests. MBPP tasks have assert statements; sampled
  programs are executed and counted correct when they pass the tests. The paper
  also creates challenge tests for a subset to detect shallow overfitting.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Code samples and prompt configurations. The paper samples many programs per
  task, compares few-shot prompt examples, and studies dialog/human feedback as
  a repair signal. It does not define an evolutionary archive.

- What is the population/database/selection mechanism?
  The sample pool is per-task. The main metric is whether any sampled program
  solves a task and the fraction of all samples that solve tasks. Selection is
  mostly oracle/test-based for analysis, not a deployment ranker.

- What evidence matters?
  The largest model solves 59.6 percent of MBPP tasks with few-shot prompting
  and 80 samples per task. Fine-tuning gives about a 10 percentage point boost
  across most model sizes. On MathQA-Python, the 137B model gets 33.4 percent
  few-shot accuracy, while fine-tuning reaches 83.8 percent on the best
  MathQA-DSL setup. Human dialog raises a 50-task edited-MBPP solve rate from
  30 percent to over 65 percent with four feedback turns.

- What does this assume that DynaWorld does not yet have?
  Clean short tasks with known signatures and cheap tests. DynaWorld has long
  trainer code, evolving research objectives, expensive render/training checks,
  and hidden state. We need to carve out MBPP-like microlib tasks before
  interpreting sample-based results.

## Datasets

The paper introduces two datasets.

MBPP:

```text
name:
    Mostly Basic Programming Problems

size:
    974 short Python tasks

task shape:
    natural-language description
    self-contained Python function
    three assert tests
    reference solution

intended difficulty:
    entry-level programmer
```

The official MBPP README gives the split:

```text
task IDs 1-10:
    few-shot prompting examples

task IDs 11-510:
    test set

task IDs 511-600:
    validation

task IDs 601-974:
    training
```

The paper notes the 974 tasks include:

```text
58 percent mathematical
43 percent list processing
19 percent string processing
9 percent integer sequences
2 percent other data structures
```

Average reference solution length is 6.8 lines, median 5, maximum 50.

The authors also create an edited/hand-verified subset:

```text
426 hand-verified questions
standard function signatures
unambiguous wording
tests aligned with text description
```

MathQA-Python:

```text
source:
    MathQA

shape:
    natural-language math word problem
    translated Python program / DSL solution

size in abstract:
    23,914 problems

character:
    more complex language
    mostly straight-line code
```

Local implication:
    DynaWorld needs two early task sets:

```text
MBPP-like microlib tasks:
    clear small functions
    cheap hidden tests
    pass@k and ranker-gap evaluation

repo-like patch tasks:
    SWE-bench style patch/evaluator harness
```

Do not jump straight to repo tasks without first proving the sampling and
ranking machinery on MBPP-like tasks.

## Prompt And Evaluation Setup

The paper evaluates:

```text
few-shot prompting:
    held-out examples concatenated into prompt

fine-tuning:
    MBPP fine-tuning on 374 examples
    MathQA-Python fine-tuning on larger dataset
```

For MBPP synthesis:

```text
temperature:
    0.5

samples per problem:
    80

correctness:
    sampled code passes tests
```

The MBPP README prompt shape:

```text
You are an expert Python programmer, and here is your task: {prompt}
Your code should pass these tests:

{tests}
[BEGIN]
{code}
[DONE]
```

Local transfer:
    The first DynaWorld microlib prompts should use a stable delimiter and an
    explicit output contract. For patch tasks, replace `[BEGIN] code [DONE]`
    with a search/replace or unified-diff schema.

## Scaling Results

The paper evaluates models from 244M to 137B parameters and finds that synthesis
performance scales approximately linearly with log model size.

MBPP headline:

```text
largest model, few-shot:
    59.6 percent of MBPP problems solved by any of 80 samples

fine-tuning:
    roughly +10 percentage points across most model sizes
```

The paper distinguishes two metrics:

```text
fraction of tasks solved by any sample:
    similar to pass@k / oracle best-in-sample
    scales predictably with model size

fraction of samples that solve their task:
    sample reliability
    improves with model size but less predictably
```

Local transfer:
    Always separate:

```text
coverage:
    does the sample pool contain any good candidate?

reliability:
    how often does a random/selected candidate work?
```

An evolver can have high coverage and poor reliability. That means ranking or
selection is the bottleneck.

## Prompt Sensitivity

The paper finds performance is not very sensitive to the number of visible
assert tests in the prompt. Including all three asserts solved only three extra
problems compared with one assert in one analysis.

But performance is highly sensitive to which few-shot examples are used:

```text
best prompt seed:
    around 60 percent tasks solved

many other prompt seeds:
    far fewer tasks solved
```

Qualitative failure:

```text
bad prompt examples:
    long repetitive samples
    examples leak irrelevant data structures/patterns
    context window issues

good prompt examples:
    short compact examples
    useful external library usage
```

Local implication:
    Prompt-example selection is itself a search dimension:

```text
prompt_example_pool
prompt_seed
example_order
example_length
example_style
```

For `alpha_evolve`, this maps to parent prompt selection. Do not treat the
prompt template as fixed. Evolve and ablate prompt examples separately from code
patches.

## Visible Tests And Challenge Tests

The paper includes an important overfitting check. For an MBPP task asking to
remove the first and last occurrence of a character, all samples from the best
model passed the three normal tests, but the tests did not cover strings with
more than two occurrences. The model learned a wrong simpler behavior: delete
all occurrences.

The authors sampled 50 of 500 test programs and wrote adversarial challenge
tests:

```text
normal tests passed:
    33 / 50 tasks

normal plus challenge tests passed:
    29 / 50 tasks

estimated over-credit:
    about 12 percent of counted solutions fail challenge tests
```

Local implication:
    Visible tests are not enough. Every local microlib task should separate:

```text
visible tests:
    given to Codex / used for cheap ranking

hidden challenge tests:
    withheld promotion gate

adversarial tests:
    generated or hand-authored to catch narrow solutions
```

This directly reinforces Agentless and CodeT. Generated tests can help, but
promotion needs repo-owned hidden gates.

## Error Modes

The paper's error breakdown shows:

```text
small models:
    type/syntax/runtime errors are common

larger models:
    syntax/runtime errors drop
    most failures become assertion failures
```

For the largest model, over 63 percent of failures are due to failing tests,
not syntax/runtime issues.

Local implication:
    For modern Codex runs, syntax guards are necessary but probably not the main
    bottleneck. The hard failures are semantic:

```text
wrong edge cases
wrong interpretation
overfit visible tests
wrong algorithm
wrong scope
```

The failure classifier should preserve this distinction.

## Human Feedback

The paper runs a small human-model dialog experiment over 50 edited MBPP tasks.
Participants can give one-sentence hints/corrections, not large code snippets,
for up to four turns.

Results:

```text
no human, one call:
    30 percent solve rate

one human feedback turn:
    55 percent solve rate

four feedback turns:
    over 65 percent solve rate
```

It also reports that human feedback solves 10 problems that the model could not
solve without assistance.

Local transfer:
    In an automated system, evaluator feedback and failure reflections are the
    substitute for human hints:

```text
failed visible test:
    one-sentence failure summary

hidden failure after promotion check:
    do not expose exact hidden data
    expose failure label / counterexample class when safe

human note:
    can be injected as curated reflection for repeated failure class
```

This supports Reflexion/Self-Refine, but with a warning: feedback quality
matters. A bad hint can steer the model wrong.

## Execution Prediction

The paper explores whether models understand execution by asking them to predict
program outputs. Even the best models generally struggle to predict execution
results for a given input, whether few-shot or fine-tuned.

Local implication:
    Do not ask Codex to internally simulate evaluator outcomes and trust it.
    Run the code.

This matters for alpha evolution:

```text
model critique:
    useful hypothesis

actual evaluator:
    source of truth
```

A candidate should never be promoted because a model says it will pass.

## MathQA-Python

MathQA-Python tests synthesis from more complex natural language, but mostly
straight-line code.

Results:

```text
137B few-shot on MathQA-Python:
    33.4 percent accuracy

best fine-tuned model on MathQA-DSL:
    83.8 percent accuracy
```

The paper also shows that hints can sharply increase sample correctness on
specific harder MathQA problems.

Local implication:
    Fine-tuning can matter when the task distribution is narrow and many
    examples are available. For DynaWorld we probably do not have enough curated
    task examples for model fine-tuning initially. The practical substitute is:

```text
prompt/example curation
retrieval over prior solved tasks
reflections
sample-and-rank
task-specific microlibs
```

## What Transfers To `codex exec`

This paper suggests the first local `codex exec` harness should look like MBPP:

```text
task.jsonl:
    task_id
    text
    visible_tests
    hidden_tests
    setup_code
    expected_output_contract

prompt:
    stable few-shot examples
    task text
    visible tests
    strict output delimiter/schema

run:
    sample N completions or patches
    execute visible tests
    execute hidden tests offline
    report pass@k, selected pass@1, ranker gap
```

The runner should support prompt-example ablations:

```text
same tasks
same model
different few-shot seeds
compare coverage/reliability
```

It should also support challenge tests:

```text
visible pass:
    not enough

hidden challenge pass:
    promotion candidate
```

## Microlibs Suggested By This Paper

```text
mbpp_like_task_schema/
    JSONL schema for small local tasks with visible and hidden tests.

prompt_example_selector/
    Selects, orders, and ablates few-shot examples.

challenge_test_registry/
    Stores hand-written or generated adversarial edge tests.

sample_reliability_metrics/
    Separates any-sample coverage from fraction-of-samples correctness.

semantic_failure_classifier/
    Distinguishes syntax/runtime/assertion/edge-case failures.

dialog_feedback_adapter/
    Converts evaluator failures into concise hints for refinement attempts.

execution_truth_gate/
    Prevents model self-evaluation from replacing actual execution.
```

## Target Problems In This Repo

Good MBPP-like tasks for DynaWorld:

```text
bounded_context_window:
    text says how to slice file context
    visible tests cover normal windows
    hidden tests cover boundaries and empty files

pass_at_k_estimator:
    visible tests cover simple n/c/k
    hidden tests cover missing samples and edge cases

patch_normalizer:
    visible tests cover whitespace/comment normalization
    hidden tests cover semantically distinct patches

metric_log_parser:
    visible tests cover typical evaluator output
    hidden tests cover missing/malformed metrics

failure_labeler:
    visible tests cover guard/test/timeout labels
    hidden tests cover combined failures
```

These can be solved as self-contained functions first, then promoted into repo
microlibs after they pass hidden tests.

## Red-Team Notes

Risk: Three visible tests are weak.
    MBPP shows normal tests can over-credit shallow solutions. Always add
    hidden challenge tests.

Risk: Prompt seed dominates.
    If one few-shot seed works and another fails, the "model" result is really
    a prompt-selection result. Track prompt IDs.

Risk: pass@k hides low reliability.
    A task solved by 1 of 80 samples is not robust. Report sample reliability.

Risk: execution prediction is weak.
    Models can explain code and still fail to predict execution. Run evaluators.

Risk: fine-tuning lesson may not transfer.
    MBPP fine-tuning uses a curated distribution. DynaWorld lacks that dataset
    initially.

Risk: edited datasets change difficulty.
    Curation improved clarity and performance. Local task quality directly
    changes measured solver quality.

## Local Falsification Tests

1. Prompt seed ablation:

```text
same 20 local microlib tasks
same model
5 few-shot example sets
measure pass@1, pass@k, reliability
```

2. Visible versus challenge tests:

```text
count candidates passing visible tests
count candidates passing hidden challenge tests
estimate visible over-credit
```

3. Feedback value:

```text
initial candidate fails
feed one concise evaluator hint
sample repair
compare against simply sampling another fresh candidate
```

4. Semantic failure classifier:

```text
syntax error
runtime error
assertion failure
edge-case hidden failure
timeout
```

Verify each produces a different failure label.

5. Model self-evaluation trap:

```text
ask Codex whether candidate will pass
run actual tests
measure calibration
```

Do not use self-evaluation as promotion signal unless calibrated.

## Design Consequences

After this paper, the first task suite should be:

```text
alpha_evolve/tasks/mbpp_like/*.jsonl
```

with fields:

```text
task_id
description
function_name_or_patch_scope
setup_code
visible_tests
hidden_tests
challenge_tests
few_shot_group
tags
```

The first experiments should compare:

```text
prompt seeds
sample counts
visible-only ranking
hidden oracle
feedback/refinement
```

Only after those metrics are stable should the runner graduate to full repo
patch tasks.

## Open Questions For Later Papers

- Does AlphaCode's filtering pipeline solve the sample-reliability problem at
  much larger scale?
- Does CodeT provide a better generated challenge-test mechanism for MBPP-like
  tasks?
- How should local few-shot examples be selected: manually, by similarity, by
  diversity, or by evolution?
- Can a small MBPP-like DynaWorld suite predict performance on real repo
  cleanup patches?
