# 020 - CodeT: Code Generation with Generated Tests

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2207.10397
    https://arxiv.org/pdf/2207.10397
    https://github.com/microsoft/CodeT
    https://github.com/microsoft/CodeT/tree/main/CodeT

Implementation artifacts inspected:
    https://github.com/microsoft/CodeT/blob/main/CodeT/README.md
    Repository structure described in the README:
    `main.py`, `src/postprocess.py`, `src/_execution.py`, `src/execution.py`,
    `src/agreement.py`, `src/evaluation.py`, `src/io_utils.py`, datasets, and
    generated data artifacts.

Bibliographic metadata:
    Authors: Bei Chen, Fengji Zhang, Anh Nguyen, Daoguang Zan, Zeqi Lin,
    Jian-Guang Lou, Weizhu Chen.
    First arXiv submission: 2022-07-21.
    arXiv v2 inspected: 2022-11-23.

Why this paper matters for alpha_evolve:
    CodeT is the generated-test counterpart to AlphaCode. AlphaCode uses
    generated inputs mostly to cluster candidate programs by output behavior.
    CodeT goes further: it asks the model to generate executable assert-style
    tests, runs every candidate against those tests, builds consensus sets, and
    ranks a solution by dual agreement with both tests and sibling solutions.

    For `alpha_evolve`, this paper gives the first concrete selection algorithm
    we can implement cheaply:

```text
generate k Codex candidate patches
generate m candidate-visible or selector-only tests/probes
execute each candidate against each test/probe
group candidates by pass/fail vector
score each group by both test support and candidate support
submit the best representative to hidden repo gates
```

    The key correction to Paper 019 is that largest behavior cluster is not
    enough. Trivial wrong code can form a large cluster. A selector should also
    ask whether that cluster passes many generated tests.

One-sentence mechanism:
    Use the same pretrained language model to sample code candidates and
    assert-style test cases, execute all candidate/test pairs, form consensus
    sets from candidates that pass the same tests, and select solutions using a
    score that combines number of supporting tests with number of agreeing code
    samples.

## Reading Questions

- What is the executable feedback signal?
  Generated tests. The paper creates assert-style tests from the problem
  description, executes candidate code on them, and uses the pass/fail matrix
  for selection. Ground-truth benchmark tests are only used for final
  evaluation.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Code samples and test samples. CodeT does not search over tool trajectories or
  mutate a persistent archive. It generates both sides of an executable
  agreement matrix.

- What is the population/database/selection mechanism?
  Per-problem sampled code candidates and per-problem sampled test cases. The
  selector groups code candidates by the exact set of generated tests they pass,
  scores each consensus set, and chooses the top group.

- What evidence matters?
  CodeT improves pass@1 across HumanEval, MBPP, APPS, and CodeContests in the
  paper's zero-shot setting. With code-davinci-002, HumanEval pass@1 rises from
  47.0 percent to 65.8 percent, MBPP from 58.1 percent to 67.7 percent, APPS
  introductory from 27.2 percent to 34.6 percent, and CodeContests from 0.7
  percent to 2.1 percent. Gains are much smaller on harder competition-level
  tasks, where generated tests are weaker.

- What does this assume that DynaWorld does not yet have?
  A way to generate runnable tests from a spec, a sandbox for executing each
  candidate/test pair, and a task format where generated tests can be validated
  or at least audited. DynaWorld must keep hidden gates separate because
  generated tests can be toxic or collusive.

## Problem Setting

CodeT targets selection among many generated code samples. The motivating gap
is the familiar pass@k gap:

```text
pass@100 high:
    one of many generated samples often works

pass@1 low:
    the model's first or randomly chosen sample often fails

selection problem:
    choose the correct sample without using the hidden benchmark tests
```

The paper's proposal:

```text
1. Generate code solutions from the task prompt.
2. Generate tests from the same task prompt using the same model.
3. Execute every code solution on every generated test.
4. Use agreement among code and tests to choose a solution.
```

For local `alpha_evolve`, the important reframe is that tests themselves can be
part of the generated population. But generated tests are not the final truth.
They are selector evidence.

## Test Case Generation

CodeT uses the same model for two roles:

```text
code generation:
    context c -> code samples X

test generation:
    concat(context c, instruction p) -> test samples Y
```

For HumanEval-style function tasks, the test-generation instruction includes:

```text
function body placeholder:
    pass

intent comment:
    check the correctness of entry point

assert prefix:
    starts the expected assertion format
```

The model then completes with assertion statements. The paper removes real
example input-output cases from the context in its main experiments to avoid
leaking benchmark tests and to make generated tests genuinely independent.

Post-processing:

```text
keep syntactically valid assertion statements
require the assertion to mention the entry-point function
limit number of valid tests extracted from each test-generation sample
```

Implementation details from the paper:

```text
temperature:
    0.8

top_p:
    0.95

max generation length:
    300

execution timeout per test:
    0.1 seconds

test-generation samples:
    100 for HumanEval and MBPP
    50 for APPS and CodeContests

test_case_limit:
    default 5 syntactically correct tests per sample
```

The README exposes this as a script interface:

```text
main.py
    source_path_for_solution
    predict_path_for_solution
    source_path_for_test
    predict_path_for_test
    cache_dir
    timeout
    test_case_limit
```

This maps cleanly to DynaWorld:

```text
task_spec.json
candidate_patch_samples.jsonl
generated_probe_samples.jsonl
execution_cache/
agreement_report.json
hidden_gate_report.json
```

## Dual Execution Agreement

Let:

```text
X:
    generated code solutions x_1 ... x_N

Y:
    generated test cases y_1 ... y_M
```

A candidate passes a test if:

```text
candidate runs without error
candidate output matches generated expected output
```

CodeT groups candidates by the exact set of generated tests they pass.

For a consensus set:

```text
Sx:
    code solutions in the group

Sy:
    generated tests passed by those solutions
```

Basic score:

```text
f(S) = |Sx| * |Sy|
```

Practical score:

```text
f(S) = sqrt(|Sx|) * |Sy|
```

The square-root weighting reduces the risk that many agreeing code samples
overpower a smaller group that passes more tests. The paper finds this more
robust than a linear count of candidate siblings.

Interpretation:

```text
|Sy|:
    test support

|Sx|:
    sibling solution support

dual agreement:
    a solution is stronger when it passes many generated tests and other
    independent samples behave the same way
```

This is different from pure AlphaCode clustering:

```text
AlphaCode-style cluster size:
    many candidates behave alike

CodeT consensus:
    many candidates behave alike and pass many generated tests
```

The second term matters because trivial wrong functions can be popular.

## Relationship To RANSAC

The paper explicitly frames CodeT as inspired by RANSAC:

```text
hypothesis:
    a candidate/test pair describes a plausible behavior

inliers:
    other candidate/test pairs agreeing with that behavior

consensus set:
    candidates and tests mutually supporting the same behavior
```

The assumptions:

```text
1. Code solutions and tests are sampled independently from the model given the
   same problem.

2. Incorrect solutions are diverse enough that independent wrong solutions are
   unlikely to agree on many tests by chance.
```

The second assumption is fragile. It often holds on short HumanEval/MBPP tasks,
but it can fail when:

```text
many candidates share the same misunderstood spec
many generated tests encode the same misunderstanding
trivial constant/identity solutions pass weak generated tests
problem statement is ambiguous
required imports or environment details are missing
```

DynaWorld should treat those assumptions as hypotheses to measure, not axioms.

## Benchmarks

The paper evaluates four benchmarks:

```text
HumanEval:
    164 hand-written Python function tasks
    average 7.77 ground-truth tests per problem
    100 code samples per problem

MBPP:
    427 sanitized tasks in the paper's setup
    average 3.1 ground-truth tests per problem
    100 code samples per problem

APPS:
    5000 problems across introductory, interview, competition levels
    average 20.99 ground-truth tests per problem
    50 code samples per problem

CodeContests:
    165 problems in the paper's setup
    average 203.7 ground-truth tests per problem
    1000 code samples per problem
```

Models:

```text
Codex:
    code-cushman-001
    code-davinci-001
    code-davinci-002

open models:
    InCoder 6.7B
    CodeGen Mono 16B
```

The paper uses a zero-shot setup and removes real examples from prompts for
main comparisons. Appendix experiments consider one-shot APPS/CodeContests,
where example tests are used as formatting hints and a public filter.

## Results

HumanEval/MBPP:

```text
code-davinci-002 HumanEval baseline pass@1:
    47.0 percent

code-davinci-002 HumanEval CodeT pass@1:
    65.8 percent

absolute improvement:
    18.8 percentage points

code-davinci-002 MBPP baseline pass@1:
    58.1 percent

code-davinci-002 MBPP CodeT pass@1:
    67.7 percent

absolute improvement:
    9.6 percentage points
```

APPS/CodeContests with code-davinci-002:

```text
APPS introductory baseline pass@1:
    27.2 percent

APPS introductory CodeT pass@1:
    34.6 percent

APPS interview baseline pass@1:
    5.1 percent

APPS interview CodeT pass@1:
    8.1 percent

APPS competition baseline pass@1:
    1.8 percent

APPS competition CodeT pass@1:
    2.2 percent

CodeContests baseline pass@1:
    0.7 percent

CodeContests CodeT pass@1:
    2.1 percent
```

The pattern is clear:

```text
shorter function-level tasks:
    strong gains

harder competitive-programming tasks:
    smaller gains

reason:
    generated tests and generated code both degrade when the model does not
    understand the problem
```

For DynaWorld, this strongly supports starting with function-level microlibs,
not full research tasks.

## Test Quality Analysis

The paper introduces two useful generated-test quality concepts.

Accuracy:

```text
generated test is correct if the canonical solution passes it
```

Toxicity:

```text
generated test is toxic if some generated candidate passes it while the
canonical solution does not
```

This is an important term for local use. Generated tests can actively mislead a
selector. They are not just noisy; they can prefer wrong behavior.

The paper finds:

```text
Codex-generated tests:
    higher accuracy
    lower toxicity
    higher coverage

InCoder/CodeGen-generated tests:
    weaker in this setup

test quality:
    strongly correlated with CodeT gains
```

HumanEval generated-test counts:

```text
code-cushman-001:
    average 410.7 extracted tests
    median 429

code-davinci-001:
    average 381.9
    median 388

code-davinci-002:
    average 391.1
    median 402

InCoder:
    average 390.1
    median 400

CodeGen:
    average 55.6
    median 42
```

HumanEval generated-test coverage:

```text
Codex and InCoder:
    over 94 percent statement and branch coverage on average

CodeGen:
    around 78 percent in the reported table
```

Coverage is not correctness. But coverage is still a useful smoke metric for
generated probes.

## Fewer Tests Ablation

The paper studies how many generated tests are needed. In HumanEval, CodeT
generates up to:

```text
100 test-generation samples * 5 valid tests per sample = 500 tests
```

But the gain persists with fewer tests:

```text
only 10 tests per problem:
    still improves code-davinci-002 pass@1 by 9.5 percentage points

performance:
    generally improves with more tests

diminishing returns:
    gap narrows around sampling number >= 50 and limit >= 3
```

Local implication:

```text
Do not overbuild test generation first.
```

For a DynaWorld microlib:

```text
3-5 hand-written visible tests
10-20 generated selector probes
3-10 hidden challenge tests
```

is enough to start measuring whether generated probes help selection. If that
does not work, hundreds of generated tests will probably only hide the failure.

## Failure Cases

CodeT fails when its two assumptions fail.

Observed failure classes:

```text
ambiguous problem statement:
    model and tests can agree on the wrong interpretation

uncovered corner case:
    correct candidate exists but generated tests miss the distinguishing case

missing imports/environment:
    candidates fail for reasons outside logical correctness

model misunderstanding:
    both code samples and test samples reflect the same wrong reading

partially correct consensus:
    a large wrong group passes many generated tests but fails a rare hidden case
```

The paper reports that for code-cushman-001 on HumanEval, 53 of 164 problems
had a correct solution generated but not ranked in the top consensus set. Manual
inspection attributes part of this to ambiguity, uncovered corner cases, and
missing imports, with many remaining failures due to model misunderstanding.

DynaWorld equivalent:

```text
ambiguous task:
    "make it faster" without a metric threshold

uncovered corner:
    visible fixture only covers F=3, hidden failure is F=32

missing environment:
    generated candidate assumes a dependency not available in uv

shared wrong reading:
    all candidates optimize a proxy metric rather than the real baseline gate

partially correct:
    patch passes parser tests but breaks project config normalization
```

The runner must store these failure classes, not just pass/fail.

## Comparison With AlphaCode

AlphaCode and CodeT both use execution and agreement, but differently:

```text
AlphaCode:
    sample code candidates
    filter by public examples
    generate inputs
    group by output behavior
    select large clusters

CodeT:
    sample code candidates
    sample tests with expected outputs
    execute candidate/test pairs
    group by pass/fail test set
    score by tests passed and sibling candidates
```

CodeT's critique of pure clustering:

```text
large cluster can be trivially wrong
```

The paper's replication of AlphaCode-style clustering can group constant or
identity-style wrong solutions into a large cluster. CodeT's test-side support
penalizes these when generated tests expose them.

Local selector design should combine both:

```text
behavioral output signatures:
    useful for diversity

generated tests:
    useful for support score

hidden gates:
    only source of promotion truth
```

Do not replace hidden gates with generated tests.

## Mapping To DynaWorld Microlibs

CodeT suggests these microlibs:

```text
generated_test_prompt_builder
    Builds test/probe-generation prompts from a task spec without including
    hidden cases.

generated_test_postprocessor
    Extracts runnable pytest/assert/function-call probes from model output.

candidate_test_matrix
    Executes each candidate against each generated test and stores pass/fail,
    timeout, stderr class, and output summary.

consensus_set_builder
    Groups candidates by pass/fail vector over generated tests.

dual_agreement_scorer
    Scores consensus sets with test support and candidate support.

test_toxicity_auditor
    Uses canonical/reference implementation or trusted hidden cases to flag
    generated tests that prefer wrong behavior.

coverage_probe_reporter
    Measures statement/branch/path coverage for generated tests when cheap.

selector_ablation_reporter
    Compares random, test-count-only, sibling-count-only, AlphaCode-style
    cluster size, and CodeT dual score.
```

Suggested folder shape:

```text
alpha_evolve/
    tests_gen/
        prompt_builder.py
        postprocess.py
        toxicity.py
        coverage.py
    selection/
        candidate_test_matrix.py
        consensus.py
        dual_agreement.py
        ablations.py
    tasks/
        dynaworld_micro/
            task_spec.py
            visible_tests.py
            hidden_tests.py
            generated_probe_prompts.py
```

## Candidate-Visible Versus Selector-Only Tests

CodeT generates tests and uses them for selection. It does not necessarily feed
all generated tests back into the code-generation prompt. For DynaWorld this
distinction should be explicit:

```text
visible tests:
    included in the candidate prompt
    fair for candidate debugging
    can be overfit

selector-only generated probes:
    generated after candidate sampling or from the same task spec
    used to compute agreement/signatures
    not pasted into candidate prompt during that generation round

hidden gates:
    hand-owned or benchmark-owned
    not generated by the same model in the same loop
    used only for promotion/evaluation
```

A safe initial contract:

```text
Candidates see:
    problem spec
    allowed files
    visible tests
    style constraints

Selector sees:
    generated probes
    candidate/test matrix
    hidden gates after selection

Future generations see:
    summarized failure labels
    not raw hidden test bodies
```

This keeps generated tests useful without turning the hidden gate into prompt
training data.

## Concrete Local Task Example

Task:

```text
jsonc_config_normalizer
```

Candidate generation:

```text
Generate an implementation for normalize_config(raw_text, base_dir) that:
    parses JSONC
    strips comments
    handles trailing commas
    expands relative paths
    rejects unknown required sections
```

Visible tests:

```text
simple comment stripping
simple trailing comma
relative path expansion
```

Generated selector probes:

```text
empty comments
nested arrays with trailing commas
paths with spaces
duplicate keys
unknown train fields
legacy optional fields
```

Candidate/test matrix:

```text
candidate_001:
    passes probes [1,2,3,5]

candidate_002:
    passes probes [1,2,3,4,5,6]

candidate_003:
    passes probes [1,2,3,4,5,6]

candidate_004:
    passes probes [1,2]
```

Consensus:

```text
group A:
    candidates 002, 003
    six probes
    likely top dual score
```

Hidden gate:

```text
run repo-owned parser fixtures
run style/LOC check
run no-env-var-fanout policy check
```

This is the smallest useful unit for testing CodeT-like selection.

## Red-Team Notes

CodeT can fail badly if copied naively.

Risks:

```text
1. Generated tests can encode the model's mistaken interpretation.

2. Generated tests can be toxic: wrong candidates pass while correct candidates
   fail.

3. Candidate and test generation from the same model are not truly independent.

4. Generated tests for repo patches may assert implementation details instead
   of user-visible behavior.

5. Generated tests can be brittle against refactors.

6. A large wrong consensus can dominate when many candidates share a shallow
   solution.

7. Hard tasks may produce low-quality tests, making CodeT worse than a public
   filter.

8. Execution cost grows as candidates times tests.

9. Test post-processing can silently discard the hard tests and keep only easy
   syntactic probes.

10. Hidden gate reuse can leak through repeated selection history.
```

Mitigations:

```text
toxicity audit:
    run generated tests against reference implementation when one exists

mutation audit:
    check whether generated tests kill known bad candidates

coverage audit:
    measure function/branch/path coverage on small microlibs

probe diversity:
    deduplicate generated tests by normalized behavior, not only string text

selector ablations:
    compare dual score against random, test-count-only, and cluster-size-only

cost cap:
    limit candidates * generated tests for first runs

hidden separation:
    never expose final hidden tests as generated-test prompt examples
```

## Falsification Tests For Local Use

Test 1: generated tests beat no generated tests.

```text
setup:
    k candidates for each microlib

selectors:
    random visible passer
    visible-test count only
    generated-test dual agreement

pass condition:
    dual agreement improves selected hidden success
```

Test 2: test-only and candidate-only scores are worse.

```text
selectors:
    |Sy| only
    |Sx| only
    sqrt(|Sx|) * |Sy|

pass condition:
    dual score beats both single-axis scores on aggregate
```

Test 3: generated-test toxicity is measurable.

```text
reference:
    trusted implementation or hidden-gate oracle

measure:
    generated tests failed by reference
    generated tests passed by known wrong candidates

pass condition:
    toxic tests are flagged and excluded or downweighted
```

Test 4: probe generation is cheaper than extra Codex candidates.

```text
compare:
    k=8 with generated probes
    k=16 without generated probes

pass condition:
    generated probes improve n@k or reduce ranker gap per dollar
```

Test 5: generated tests do not become hidden gates.

```text
check:
    selected candidates pass generated tests
    hidden gate still catches some generated-test passers

pass condition:
    report distinguishes generated-probe success from hidden promotion
```

## Implementation Order After This Paper

Recommended order:

```text
1. Task spec schema for microlibs.
2. Candidate runner using `codex exec "<prompt>"`.
3. Visible test runner.
4. Generated probe prompt builder and postprocessor.
5. Candidate/test execution matrix.
6. Consensus set builder.
7. Dual agreement scorer.
8. Hidden gate runner.
9. n@k/pass@k/ranker-gap report.
10. Toxicity and coverage audit.
```

Do not build first:

```text
self-modifying evaluator prompts
automatic hidden-test generation as promotion truth
large persistent memory
complex island scheduling
full repo repair tasks
```

This paper's lesson is smaller and sharper: make selection measurable before
making evolution elaborate.

## Connections To Earlier Papers

AlphaEvolve:
    CodeT can serve as a front-end selection operator inside an AlphaEvolve
    generation. It does not replace the program database.

FunSearch:
    Generated tests are not a numeric objective by themselves; they are a way
    to choose which candidate reaches the real evaluator.

Eureka:
    Generated reward/test code can be powerful, but it needs audits against
    trusted outcomes.

Reflexion/Self-Refine:
    Natural-language feedback is useful after failure, but CodeT shows that
    executable agreement can select before verbal reflection.

Tree of Thoughts/LATS:
    The pass/fail matrix is a better state value estimate than free-form
    self-evaluation when code is executable.

SWE-agent/OpenHands:
    The execution sandbox and logs are not incidental; CodeT requires reliable
    candidate/test execution at scale.

Agentless:
    This is another argument for a simple structured pipeline before ornate
    agents: generate, execute, agree, select.

HumanEval/MBPP:
    CodeT's strongest results are exactly on the kind of small executable tasks
    we should use for the first DynaWorld microlib suite.

AlphaCode:
    CodeT improves the cluster selection idea by using generated tests with
    expected outputs, not only generated inputs and output-equivalence clusters.

## Open Questions After Reading

- Can Codex generate selector probes that are behavior-level rather than
  implementation-detail tests?
- Should generated tests be created before candidate generation, after
  candidate generation, or both?
- Does using the same model for candidates and tests create systematic
  collusion in DynaWorld microlibs?
- How should toxic tests be downweighted when no canonical implementation
  exists?
- Is sqrt(candidate_count) the right weighting for repo patches, or should
  candidate count be capped more aggressively?
- What is the minimum generated-probe count where dual agreement beats random?
- Can generated tests be reused across generations, or do they become stale
  after parent code changes?
- How do we deduplicate generated tests by behavior when the code under test is
  a file patch rather than a pure function?

## Bottom Line

CodeT provides the first practical selector to implement:

```text
code candidates
generated tests
candidate/test matrix
consensus groups
dual agreement score
hidden-gate validation
```

For DynaWorld, generated tests should be treated as selector evidence and
diagnostic probes, not as promotion truth. The immediate target is a small
microlib harness where CodeT-style dual agreement can be compared against
random selection, visible-test filtering, and AlphaCode-style cluster size.
