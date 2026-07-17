# 019 - Competition-Level Code Generation with AlphaCode

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2203.07814
    https://arxiv.org/pdf/2203.07814
    https://www.science.org/doi/10.1126/science.abq1158
    https://www.deepmind.com/blog/competitive-programming-with-alphacode

Implementation artifacts inspected:
    https://github.com/google-deepmind/code_contests
    https://github.com/deepmind/code_contests

Bibliographic metadata:
    Authors: Yujia Li, David Choi, Junyoung Chung, Nate Kushman,
    Julian Schrittwieser, Remi Leblond, Tom Eccles, James Keeling,
    Felix Gimeno, Agustin Dal Lago, Thomas Hubert, Peter Choy,
    Cyprien de Masson d'Autume, Igor Babuschkin, Xinyun Chen,
    Po-Sen Huang, Johannes Welbl, Sven Gowal, Alexey Cherepanov,
    James Molloy, Daniel J. Mankowitz, Esme Sutherland Robson,
    Pushmeet Kohli, Nando de Freitas, Koray Kavukcuoglu, Oriol Vinyals.
    First arXiv submission: 2022-03-16 title page; arXiv v1 metadata
    is 2022-03-15/2022-03-16 depending on mirror.
    Journal version: Science, 2022.

Why this paper matters for alpha_evolve:
    AlphaCode is the clearest predecessor to an AlphaEvolve-style code search
    runner before the explicit evolutionary-program-database papers. The paper
    is not just "a bigger code model." Its useful mechanism is:

```text
sample a very large candidate pool
filter using tests visible to the candidate
build additional behavioral signatures
cluster semantically similar candidates
spend a small hidden-test submission budget on diverse representatives
measure the gap between unlimited pass@k and bounded 10@k
```

    For `alpha_evolve`, this is the missing bridge between Codex one-shot
    patching and population evolution. A local runner cannot afford one million
    Codex calls per problem, but it can copy the AlphaCode separation of
    concerns:

```text
generation budget:
    how many Codex candidates we buy

visible filter:
    cheap tests candidates are allowed to see

behavior signature:
    extra generated probes used to compare candidates

submission selector:
    choose a bounded diverse set for the real gate

hidden gate:
    repo-owned tests/benchmarks not exposed as prompt feedback
```

One-sentence mechanism:
    Train large encoder-decoder code models on GitHub and CodeContests, sample
    huge Python/C++ candidate pools per competitive-programming problem, filter
    candidates with example tests, cluster remaining candidates by behavior on
    generated test inputs, and submit at most ten diverse programs to hidden
    tests.

## Reading Questions

- What is the executable feedback signal?
  There are two levels. During candidate selection, the executable signal is
  public/example input-output tests from the problem statement plus generated
  test inputs used for behavioral clustering. During final evaluation, the
  executable signal is hidden contest tests or actual Codeforces judging.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Full program code samples. The paper also trains a separate test-input
  generator for clustering, but the main search object is still a program
  candidate sampled from a model.

- What is the population/database/selection mechanism?
  The population is a large per-problem sample pool, not a persistent evolving
  archive. Selection is a pipeline: keep candidates passing example tests,
  execute survivors on generated inputs, cluster by output behavior, then pick
  one representative from each large cluster before spending the limited
  submission budget.

- What evidence matters?
  The headline evidence is competitive performance on 10 Codeforces contests:
  average estimated top 54.3 percent ranking with a 10-submission-per-problem
  budget, corresponding to an estimated Codeforces rating of 1238 and top 28
  percent among active users in the paper's comparison set. On CodeContests,
  41B plus clustering reaches 34.2 percent 10@1M validation solve rate and
  29.6 percent 10@100K test solve rate.

- What does this assume that DynaWorld does not yet have?
  Short standalone problems, unambiguous input-output semantics, cheap
  execution, many samples per problem, and hidden tests that can judge
  correctness without human interpretation. DynaWorld needs microlibs that
  emulate this contract before applying the same shape to trainer or renderer
  code.

## Problem Setting

AlphaCode targets competitive programming, where a problem statement describes
an algorithmic task and contestants submit full programs. The task has several
properties that make it a useful but dangerous analogy for DynaWorld:

```text
input:
    long natural-language problem statement
    visible example tests
    optional metadata such as language, rating, tags

output:
    complete C++ or Python program

public feedback:
    example tests from the prompt

private feedback:
    hidden judge tests and resource limits

budget:
    small number of official submissions
```

The competitive-programming format forces a distinction that many agent loops
blur:

```text
candidate-visible score:
    example tests pass/fail

selection-only score:
    generated behavioral probes used to cluster/rank

final truth score:
    hidden tests or live judge result
```

For `alpha_evolve`, this distinction should become a hard interface boundary.
Candidates can receive visible test failures as prompt feedback. They should
not see the hidden gate directly, except through a promotion/no-promotion
decision recorded by the runner.

## CodeContests Dataset

The paper introduces CodeContests, a competitive-programming dataset with:

```text
sources:
    Codeforces
    Description2Code
    CodeNet

languages:
    C++
    Python
    Java

per-problem contents:
    natural-language statement
    example tests
    hidden tests where available
    generated tests from mutation
    correct human submissions
    incorrect human submissions
```

The public repository exposes `ContestProblem` protocol buffers in Riegeli
format, with train/validation/test splits and execution/evaluation code. The
README notes a full dataset download of roughly 3 GiB from Google Cloud
Storage.

Important split detail:

```text
GitHub pretraining snapshot:
    2021-07-14

training data:
    publicly released on or before 2021-07-14

validation data:
    2021-07-15 to 2021-09-20

test data:
    after the validation period
```

This temporal split matters. It is the same kind of rule DynaWorld needs for
generated benchmarks: a candidate should not be evaluated on tasks whose exact
solution or fixture has leaked into its prompt history.

The dataset also includes incorrect submissions. AlphaCode uses them as a
training signal rather than throwing them away, through value conditioning and
value prediction. That is useful for local design because failed Codex attempts
are also valuable data if they are labeled and stored cleanly.

## Dataset Quality And False Positives

The paper spends real effort on false positives. In programming tasks, a weak
test suite can accept an incorrect program. A program can also be logically
right on shown tests but too slow for actual contest constraints.

The authors reduce false positives by generating additional tests:

```text
source:
    existing test inputs

mutations:
    bit flips for binary inputs
    integer increment/decrement
    string character swaps/changes

filter:
    run 30 correct solutions
    keep generated test only when all agree on output

cap:
    up to 10 CPU hours or 200 generated tests per problem
```

Validation/test problems are kept only when test coverage is not trivially weak:

```text
minimum hidden/generated tests:
    5

minimum distinct outputs:
    2
```

Reported effect:

```text
CodeContests raw false positive rate:
    high; paper reports 62 percent before generated tests/filtering

CodeContests after generated tests/filtering:
    roughly 4 percent false positive rate in the manual estimate
```

The DynaWorld transfer is direct: before we trust an `alpha_evolve` benchmark,
we need a false-positive audit. A microlib should not be promoted just because
one visible smoke test passes. The benchmark task itself needs challenge cases,
distinct-output checks, and a record of what candidate shortcuts it is meant to
catch.

## Model Architecture

AlphaCode models the task as sequence-to-sequence translation:

```text
encoder input:
    natural-language problem description plus metadata

decoder output:
    program tokens

conditional distribution:
    p(solution | problem)
```

Architecture choices:

```text
model family:
    encoder-decoder Transformer

sizes:
    300M
    1B
    3B
    9B
    41B

encoder sequence length:
    1536 tokens

decoder sequence length:
    768 tokens

tokenizer:
    SentencePiece, 8000-token vocabulary

attention:
    multi-query attention to reduce sampling memory/cache cost
```

The asymmetric encoder/decoder matters because problem statements are longer
than solutions. The multi-query attention matters because AlphaCode's result is
heavily sampling-driven; cheaper sampling translates directly into more
candidates under the same compute budget.

DynaWorld should not copy this architecture. Codex is the model. But the design
lesson still applies: every token and context choice should be judged by its
effect on candidate quality per dollar and per wall-clock minute. If a prompt
section does not raise the slope of useful candidates, it is not free.

## Pretraining And Fine-Tuning

Pretraining:

```text
data:
    GitHub code

decoder loss:
    next-token prediction

encoder loss:
    masked language modeling

file split:
    content before a sampled pivot feeds encoder
    content after the pivot feeds decoder
```

Fine-tuning:

```text
data:
    CodeContests

input:
    problem description

output:
    solution program

losses:
    decoder next-token prediction
    encoder masked language modeling

extra mechanisms:
    tempering
    metadata conditioning
    value conditioning
    value prediction during training
    GOLD-style objective stage
```

The extra mechanisms are less important for a Codex runner than the fact that
the system trains on both success and failure labels.

For local `alpha_evolve`, we can simulate this without training:

```text
candidate record:
    prompt_hash
    parent_id
    diff
    visible_test_result
    generated_probe_signature
    hidden_gate_result
    failure_label
    evaluator_version

future prompt context:
    include selected failure summaries
    condition next candidates on "must fix class X failure"
```

That is the inference-time analogue of value conditioning: future generations
know what kind of past output was wrong, without requiring weight updates.

## Value Conditioning And Failure Data

AlphaCode inserts correctness metadata into problem descriptions during
fine-tuning, so the model sees whether a training submission is correct or
incorrect. At sampling time, it conditions on the desired label: correct
solution.

The paper also adds value prediction during training, using token
representations to classify whether a submission is correct. This auxiliary
prediction is not used during sampling, but it provides an extra supervision
signal.

Local transfer:

```text
do not throw away failed Codex patches

store failure class:
    syntax_error
    import_error
    visible_test_fail
    hidden_test_fail
    timeout
    metric_regression
    line_count_bloat
    user_contract_violation

store failure surface:
    exact command
    normalized stderr tail
    failing test node
    affected file list
    evaluator fingerprint

use failure data:
    prompt future candidates
    train or tune rankers later
    detect repeated traps
```

This is not "reflection" as free-form journaling. It is structured negative
data with enough metadata to keep failures comparable.

## GOLD And Precision Over Recall

The paper frames program synthesis as a one-of-many problem. Many distinct
programs can solve a problem. Standard maximum likelihood training spreads
probability over many solutions in the dataset, while contest success only
requires finding one correct program within a small submission budget.

GOLD is used to bias learning toward tokens already in the model's distribution
and away from trying to cover every possible solution style. The exact training
algorithm is less relevant locally because we are not fine-tuning Codex here.

The local lesson is about selection pressure:

```text
wrong objective:
    make every generated patch look like a human reference patch

better objective:
    find one patch that passes immutable repo gates and stays maintainable

wrong metric:
    average similarity to references

better metric:
    bounded-submission solve rate under hidden gates
```

This supports evaluating `alpha_evolve` with `n@k`-style metrics, not just
single-run success.

## Large-Scale Sampling

AlphaCode's largest practical lever is sample count. The paper samples up to
millions of programs per problem. Sampling is parallelizable, so a stronger
system can buy search by spending accelerators.

Diversity tricks used during sampling:

```text
language split:
    half Python
    half C++

metadata randomization:
    random problem tags
    random ratings from 800 to 3500

temperature:
    relatively high
    tuned against sample budget
```

Important result:

```text
low sample budget:
    lower temperatures can be better

high sample budget:
    higher temperatures can be better

top-k and nucleus:
    did not significantly outperform temperature sampling in the reported
    AlphaCode sweeps
```

For Codex-driven evolution, the analogue is not temperature alone. We need
explicit diversity axes:

```text
prompt family:
    direct patch
    minimal diff
    test-first
    performance-first
    reviewer-style repair

search target:
    function replacement
    config normalizer
    evaluator implementation
    generated-test suite

risk profile:
    conservative local edit
    bolder refactor
    alternate algorithm

context slice:
    only target function
    target plus callers
    docs plus tests plus failure log
```

Because Codex calls are expensive compared with Transformer sampling, local
`alpha_evolve` should start with small budgets:

```text
k = 4:
    cheapest one-shot baseline

k = 8:
    enough to test selection diversity

k = 16:
    enough to measure ranker gap on microlibs

k >= 32:
    only after evaluator cost and replay are stable
```

## Filtering

The paper's filtering stage keeps only samples that pass example tests from the
problem statement. This removes roughly 99 percent of model samples. Even after
that, many problems still have thousands of survivors.

Key observation:

```text
filtering is necessary but not sufficient
```

Filtering answers:

```text
could this candidate satisfy the obvious public contract?
```

It does not answer:

```text
is this candidate algorithmically correct?
is this candidate efficient?
does it handle hidden corner cases?
is it different from the other survivors?
```

Local transfer:

```text
visible filter:
    pytest node or tiny smoke visible in prompt

hidden gate:
    fuller test suite or benchmark not pasted into prompt

expected visible-filter pass rate:
    low enough to reject weak samples
    high enough to leave multiple candidates

bad visible filter:
    either every sample passes or no sample passes
```

For DynaWorld, a visible filter should be cheap and narrow:

```text
example:
    run a deterministic parser test
    execute a fixture-level metric check
    validate a generated config schema
    compare behavior signatures on a tiny tensor
```

It should not be a long training run. Expensive metrics belong after candidate
selection.

## Clustering

After filtering, AlphaCode clusters candidates by behavior. The system executes
each surviving program on generated test inputs and groups programs that
produce the same outputs. The intuition is that syntactically different
programs can be semantically identical.

Test-input generator:

```text
architecture:
    same as main AlphaCode models

initialization:
    same GitHub pretrained checkpoint

training target:
    test inputs from problem descriptions

training data:
    example tests
    hidden tests
    generated tests

test-time role:
    create additional inputs for unseen problems
```

The generated inputs do not need to be perfectly valid. Even imperfect inputs
can help group candidates by behavior if they expose output differences.

Cluster selection rule:

```text
1. execute filtered candidates on generated inputs
2. group candidates by output signature
3. sort clusters from largest to smallest
4. take one solution from each cluster
5. if fewer than budget clusters exist, cycle back and skip already submitted
   samples
```

The surprising detail is that larger clusters worked best. The paper suggests
many wrong programs fail in many different ways, while correct programs tend to
agree with each other and therefore form a larger behavioral cluster.

Local transfer:

```text
behavior signature for parser microlib:
    outputs on generated JSONC/config examples

behavior signature for metric microlib:
    numeric outputs on small synthetic tensors

behavior signature for patch selector:
    chosen file/line spans over issue fixtures

behavior signature for evaluator generator:
    pass/fail vector over candidate patches

behavior signature for benchmark harness:
    normalized command result over fake worktrees
```

Do not cluster by diff text. Cluster by externally observed behavior.

## Metrics

The paper defines `n@k`:

```text
k:
    number of generated samples

n:
    number of samples selected for hidden-test evaluation

n@k:
    percentage of problems solved when only n of k samples can be submitted
```

The main contest-like metric is:

```text
10@k:
    ten submissions chosen from k generated candidates
```

It also reports:

```text
pass@k:
    hidden-test success if all k generated candidates could be evaluated
    equivalent to k@k
    upper bound on selection with budget n < k
```

This distinction is critical for local evaluation:

```text
pass@k high, n@k low:
    generator can produce a solution, selector/ranker is weak

pass@k low, n@k low:
    generator/context/problem setup is weak

visible-filter pass high, hidden success low:
    public tests are under-specified

visible-filter pass low, hidden success exists:
    public tests may be too strict or misaligned
```

For the first `alpha_evolve` runner, each task report should include:

```text
k:
    number of Codex candidates

n:
    hidden submissions allowed

visible_pass_count:
    candidates passing public filter

cluster_count:
    behavior clusters among visible passers

hidden_success:
    whether selected candidate passed hidden gate

oracle_success:
    whether any candidate in the full pool passed hidden gate

ranker_gap:
    oracle_success - selected_success
```

## Results

Codeforces:

```text
evaluation window:
    contests from 2021-12-01 to 2021-12-28

contest filter:
    more than 5000 participants

number of contests:
    10

system:
    ensemble of 41B and 9B with clustering for reported contest run

average estimated ranking:
    top 54.3 percent, where lower percentile means more users above AlphaCode

estimated Codeforces rating:
    1238

comparison:
    within top 28 percent of users active in the previous 6 months
```

CodeContests validation/test:

```text
41B validation 10@1M:
    31.8 percent

41B plus clustering validation 10@1M:
    34.2 percent

41B plus clustering validation 10@100K:
    31.8 percent

41B plus clustering test 10@100K:
    29.6 percent
```

Filtering/clustering:

```text
example tests:
    remove more than 99 percent of samples

41B with one million samples:
    at least one sample passes example tests for more than 90 percent of
    validation problems

clustering:
    improves 10@k over filtering alone

pass@k:
    remains a large upper bound, showing selection is still imperfect
```

Scaling:

```text
sample count:
    solve rate grows roughly log-linearly with more samples

model size:
    larger models have better same-budget solve rates and better slopes

compute tradeoff:
    train better model versus sample more from current model
```

For DynaWorld, the scaling result should be read as:

```text
each extra Codex sample has diminishing returns
ranker quality matters more as k rises
cheap tasks are needed to estimate the sample/selector curve before expensive
renderer tasks
```

## Codeforces Simulation Details

The appendix describes a simulated live contest setup:

```text
hardware assumption:
    3750 TPUv4 chips plus 3750 TPUv4i chips

sampling:
    continuous during contest

worker pool:
    evaluates samples against example tests

submission points:
    up to three per problem
    based on contest time remaining or relative number of example-test passers

clustering time:
    120 seconds added before submission

submission cap:
    total of 10 per problem
```

The important engineering point is that selection has latency. AlphaCode did
not simply generate candidates once at the end. It coordinated sampling,
filtering, clustering, and timed submissions under contest scoring.

Local `alpha_evolve` also needs to model time and cost:

```text
candidate generation time
visible test time
clustering/probe time
hidden gate time
wall-clock budget
token budget
```

Without these fields, "best candidate wins" experiments will overestimate what
an actual autonomous runner can do.

## Relationship To AlphaEvolve

AlphaCode differs from AlphaEvolve:

```text
AlphaCode:
    one-shot per-problem sample pool
    no persistent program database
    no evolutionary mutation loop over prior code
    hidden judge tests decide success
    generated tests mainly for clustering

AlphaEvolve:
    persistent program database
    iterative LLM mutations
    evaluator cascade
    multi-objective fitness
    promotion across generations
```

But AlphaCode gives the selection substrate AlphaEvolve needs:

```text
large or modest candidate pool
visible filter
behavioral descriptors
diversity-aware selection
hidden gate separation
n@k-style evaluation
```

The local implementation should not jump straight to an island archive before
we can measure these primitives. First prove that a bounded set of Codex
candidates can be generated, filtered, behavior-clustered, and selected better
than random.

## Mapping To DynaWorld Microlibs

AlphaCode suggests these microlibs:

```text
candidate_pool_store
    Stores candidate metadata, prompts, diffs, stdout/stderr, evaluator
    fingerprints, and costs.

codex_sample_runner
    Calls `codex exec "<prompt>"` in isolated worktrees or temp copies.
    The CLI prompt path must not confuse `-p` profile flags with prompt text.

visible_test_filter
    Runs candidate-visible tests and records normalized pass/fail vectors.

generated_probe_builder
    Builds additional inputs or fixtures for behavioral comparison.

behavior_signature
    Converts candidate execution on probes into a stable comparable signature.

behavior_clusterer
    Groups candidates by signature, not by diff text.

budgeted_submission_selector
    Picks n representatives from k candidates, preferring cluster diversity
    and visible score quality.

hidden_gate_runner
    Runs repo-owned tests/benchmarks that are not pasted into candidate prompts.

false_positive_auditor
    Estimates whether visible tests accept hidden-gate failures too often.

ranker_gap_reporter
    Compares selected candidate success against oracle best-of-k success.
```

For the first implementation, keep each microlib small and file-backed:

```text
alpha_evolve/
    runners/
        codex_sample_runner.py
        command_runner.py
        worktree_runtime.py
    evaluation/
        visible_filter.py
        hidden_gate.py
        behavior_signature.py
        clustering.py
        selection.py
    tasks/
        mbpp_like/
        dynaworld_micro/
    storage/
        candidate_store.py
        event_log.py
    reports/
        nk_metrics.py
        ranker_gap.py
```

This structure is intentionally more AlphaCode than full AlphaEvolve. The
archive and mutation policy can come after the candidate/filter/cluster/gate
loop is measurable.

## Candidate Task Shapes For This Repo

Start with tasks that look like competitive-programming problems: clear input,
clear output, cheap execution, hidden tests.

Good first targets:

```text
jsonc_config_normalizer:
    Input: messy JSONC train config.
    Output: normalized config dict or error.
    Visible tests: common comments/trailing commas.
    Hidden tests: malformed keys, legacy defaults, path normalization.

metric_aggregator:
    Input: per-frame/per-view metric records.
    Output: stable aggregate summary.
    Visible tests: simple averages.
    Hidden tests: missing frames, NaNs, weighted views.

result_table_parser:
    Input: result JSONs and logs.
    Output: baseline row candidate.
    Visible tests: one clean fixture.
    Hidden tests: missing W&B id, stale result, partial run.

renderer_capability_selector:
    Input: feature dimension, device, config flags.
    Output: selected renderer backend and reason.
    Visible tests: F=3 and F=32.
    Hidden tests: unavailable .so, Metal/CUDA mismatch, feature-color path.

prompt_context_packer:
    Input: task spec plus file snippets.
    Output: prompt payload under a token budget.
    Visible tests: obvious include/exclude.
    Hidden tests: duplicated files, huge notes, required contract omission.
```

Bad first targets:

```text
full trainer rewrite:
    too large, too many hidden side effects

novel-view architecture search:
    expensive and evaluator-limited

PowerFoam full acceptance:
    important but too costly for the first pass@k loop

renderer kernel optimization:
    hard to behavior-cluster without strong benchmarks
```

## Prompting Implications

AlphaCode randomizes tags/ratings/language to increase diversity. For Codex,
diversity should be prompt-controlled and explicit:

```text
candidate families:
    minimal patch
    test-first implementation
    algorithmic rewrite
    conservative adapter
    config-only solution
    helper extraction solution

visible instructions:
    exact public tests
    allowed files
    line-count pressure
    style constraints from AGENTS.md

hidden from candidate:
    hidden fixtures
    final benchmark thresholds
    selector's cluster signatures
```

Prompt records should include:

```text
prompt_id
prompt_family
context_files
visible_tests
forbidden_files
model
reasoning_effort
codex_cli_invocation
```

This lets us later answer whether success came from more samples, better
context, better visible tests, or selector luck.

## Falsification Tests For A Local AlphaCode-Style Runner

The first local runner is only useful if it beats dumb baselines.

Test 1: selector beats random among visible passers.

```text
setup:
    generate k candidates for 10 microlib tasks

compare:
    random visible-passing candidate
    largest-cluster representative
    diversity selector
    cheap score selector

pass condition:
    selected hidden success exceeds random by a meaningful margin
```

Test 2: visible tests are not too weak.

```text
measure:
    hidden fail rate among visible passers

bad outcome:
    most visible passers fail hidden tests for the same obvious corner case

fix:
    add visible challenge case or generated probe
```

Test 3: generated probes add selection signal.

```text
compare:
    cluster by no probes
    cluster by hand-written probes
    cluster by generated probes

pass condition:
    probe clusters correlate with hidden outcomes or reduce duplicate
    submissions
```

Test 4: bounded n@k exposes ranker gap.

```text
record:
    selected_success
    oracle_success

bad outcome:
    oracle succeeds but selector fails often

fix:
    improve selector before increasing k
```

Test 5: cost curve is visible.

```text
report:
    tokens spent
    wall-clock time
    test time
    hidden gate time
    solve rate

bad outcome:
    larger k improves pass@k but not n@k

fix:
    improve selection or task probes before sampling more
```

## Red-Team Notes

AlphaCode can overstate what transfers to repo agents:

```text
1. Competitive programming has clean hidden tests.
   DynaWorld often has research-quality metrics and ambiguous success.

2. AlphaCode buys enormous sample counts.
   Codex candidates are expensive, stateful, and slower.

3. Generated tests can be invalid.
   Invalid probes may still cluster behavior, but can also reward nonsense.

4. Larger clusters are not always better.
   If many candidates share the same shallow hack, the largest cluster can be
   wrong.

5. Example tests are public under-specifications.
   A runner trained to pass them can overfit visible feedback.

6. Hidden tests can become de facto training data.
   Repeated attempts against the same hidden gate can leak the benchmark through
   selection history.

7. Full-program outputs differ from repo patches.
   Diffs interact with surrounding code, imports, configs, and stateful files.

8. Syntax and runtime failures are only the first filter.
   DynaWorld also cares about code organization, W&B logging, baselines, and
   project notes.
```

Specific DynaWorld failure mode:

```text
candidate A:
    changes code minimally, passes visible tests, fails hidden edge case

candidate B:
    rewrites helper broadly, passes hidden edge case, violates style/LOC

candidate C:
    changes config default, passes all tests, silently shifts training contract
```

AlphaCode's hidden judge would choose B or C if tests pass. DynaWorld needs
additional maintainability and contract gates.

## What To Build Next Because Of This Paper

The next `alpha_evolve` implementation should build an AlphaCode-style
selection harness before a full evolutionary archive.

Minimal slice:

```text
1. Define 5-10 MBPP-like DynaWorld microlib tasks.
2. For each task, define visible tests and hidden tests.
3. Generate k Codex candidates with prompt-family diversity.
4. Run visible tests and store normalized results.
5. Run generated or hand-written behavior probes on visible passers.
6. Cluster by behavior signature.
7. Select n candidates for hidden gates.
8. Report n@k, pass@k oracle, cluster count, visible pass rate, hidden pass
   rate, and ranker gap.
```

Do not start with:

```text
persistent islands
automatic prompt evolution
global program database
training-run optimization
large renderer benchmarks
```

Those are second-stage once the sample/filter/cluster/submission contract is
solid.

## Open Questions After Reading

- How many Codex candidates per task are enough to see a ranker gap: 4, 8, 16,
  or 32?
- Can generated tests be produced by Codex itself, or should the first probes
  be hand-written to avoid generator/evaluator collusion?
- Does largest-cluster selection still work for repo patches, or do common
  wrong patches dominate the largest cluster?
- What is the right behavior signature for file edits that change APIs but not
  immediate outputs?
- Should hidden tests be reusable across many runs, or should they rotate to
  avoid benchmark leakage?
- Can we build a false-positive audit that flags weak visible filters before
  they shape candidate behavior?
- Where should cost be charged: per prompt, per candidate, per visible test, per
  hidden gate, or per full task?
- How do we prevent a candidate from editing the visible tests themselves?

## Bottom Line

AlphaCode contributes the selection kernel for local AlphaEvolve work:

```text
generate many
filter cheaply
cluster by behavior
submit few
measure selected success against oracle best-of-k
```

For DynaWorld, this should become the first runnable benchmark harness. Only
after that harness shows that selection beats random under a hidden gate should
we add persistent evolutionary memory, island scheduling, and larger
science-code targets.
