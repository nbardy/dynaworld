# 015 - Agentless: Demystifying LLM-based Software Engineering Agents

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2407.01489
    https://arxiv.org/pdf/2407.01489
    https://github.com/OpenAutoCoder/Agentless

Implementation artifacts inspected:
    https://github.com/OpenAutoCoder/Agentless/blob/main/README.md
    https://github.com/OpenAutoCoder/Agentless/blob/main/agentless/fl/localize.py
    https://github.com/OpenAutoCoder/Agentless/blob/main/agentless/repair/repair.py

Bibliographic metadata:
    Authors: Chunqiu Steven Xia, Yinlin Deng, Soren Dunn, Lingming Zhang.
    First arXiv submission: 2024-07-01.
    Version inspected: arXiv v2, 2024-10-29.

Current implementation note:
    The repository README reports later Agentless variants beyond the paper,
    including a December 2024 Claude 3.5 Sonnet integration with higher
    SWE-bench Lite and Verified solve rates. This note focuses on the paper's
    mechanism and uses the repo only to confirm implementation shape.

Why this paper matters for alpha_evolve:
    Agentless is the necessary corrective after SWE-agent. It asks whether the
    autonomous agent loop is needed at all for repo-level repair, and shows that
    a staged, non-agentic pipeline can be highly competitive:

```text
localization
    -> repair / patch sampling
    -> patch validation and ranking
```

    For `alpha_evolve`, this says the first baseline should not be a complex
    autonomous Codex swarm. It should be a simple staged pipeline around:

```text
codex exec "<prompt>"
```

    with deterministic context construction, multiple sampled patches, test
    filtering, and patch ranking. Evolution is only useful after this baseline
    is real and measured.

One-sentence mechanism:
    Agentless resolves SWE-bench issues by hierarchically localizing likely edit
    locations, sampling many simple diff patches over those locations, generating
    reproduction tests, filtering patches with regression and reproduction test
    signals, and selecting a final patch by normalized majority/reranking, all
    without giving the LLM autonomous tool control.

## Reading Questions

- What is the executable feedback signal?
  Generated reproduction tests plus existing regression tests. Reproduction tests
  are first checked against the original repo for an issue-reproduced marker.
  Candidate patches are then filtered by regression failures and, when useful,
  by whether the generated reproduction test reports issue resolved.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Agentless searches over localized edit locations, candidate patches, and
  generated reproduction tests. It explicitly does not search over autonomous
  trajectories.

- What is the population/database/selection mechanism?
  There is no persistent evolutionary population. Within one issue, Agentless
  samples multiple edit-location sets, multiple patches per location set, and
  multiple reproduction tests. Selection is by validation/ranking, not by
  iterative agent planning.

- What evidence matters?
  The paper reports 96/300 SWE-bench Lite fixes, or 32.00 percent, at average
  cost 0.70 dollars using GPT-4o. It also reports 194/500, or 38.80 percent, on
  SWE-bench Verified. The central claim is not that agents are useless; it is
  that a simple staged method should be the baseline before claiming value from
  complex agent loops.

- What does this assume that DynaWorld does not yet have?
  Agentless assumes a SWE-bench issue statement and a repo-level patch task. For
  DynaWorld, we need small task specs, cheap tests, hidden validation, and a way
  to define correct edit locations or compare behavioral outcomes before the
  staged pipeline can be evaluated.

## Mechanism

Agentless has three phases:

```text
1. localization:
    file-level localization
    related element localization
    fine-grained edit-location localization

2. repair:
    build code context around edit locations
    sample multiple patches in simple edit/diff format

3. patch validation:
    generate reproduction tests
    identify regression tests
    filter/rank patches
    submit one final patch
```

The key difference from SWE-agent:

```text
SWE-agent:
    LLM decides next action after each observation
    multi-turn trajectory over tools
    agent can wander, loop, or recover

Agentless:
    system decides the stages
    LLM fills bounded subtasks
    no autonomous future-action decisions
    failure points are easier to inspect
```

For `alpha_evolve`, that difference is architectural. The staged form is easier
to reproduce, ablate, cache, and evolve. It also gives clearer local metrics:

```text
file localization recall
edit-location recall
patch parse success
syntax success
regression pass count
reproduction-test plausibility
final hidden score
```

## Why Agentless Avoids Agents

The paper argues that agent-based software engineering has three recurring
problems:

```text
complex tool usage/design:
    the tool API has to be designed carefully
    models can misuse complex tools
    bad tool calls waste turns and budget

lack of control in decision planning:
    agent chooses what to do next
    action space is large
    long trajectories are hard to debug

limited self-reflection:
    bad early information can be amplified
    agents do not reliably discard misleading feedback
```

Agentless does not claim that agents cannot solve tasks. It claims the
incremental value of autonomy has to beat a strong staged baseline.

Local transfer:
    The `alpha_evolve` folder should contain an `agentless_baseline` runner
    before it contains multi-agent tree search. This gives a floor for later
    claims:

```text
if staged sample-and-rank beats autonomous loop:
    keep system staged

if autonomous loop beats staged baseline:
    inspect which failures it solves

if evolution beats both:
    inspect whether it improves localization, patch diversity, or selection
```

## Localization

Agentless uses hierarchical localization because full repository context is too
large and too distracting.

Step 1: file-level localization.

```text
inputs:
    issue statement
    repo structure
    filtered source files

methods:
    prompting-based localization
    embedding-based retrieval
    irrelevant-folder filtering
    combined file set
```

From Table 2:

```text
prompting-based file localization:
    contains ground truth: 78.67 percent
    average LoC: 3,221
    cost: 0.02 dollars

embedding-based without irrelevant filtering:
    contains ground truth: 67.67 percent
    average LoC: 3,388
    cost: 0.05 dollars

embedding-based with irrelevant filtering:
    contains ground truth: 70.33 percent
    average LoC: 3,622
    cost: 0.04 dollars

combined:
    contains ground truth: 81.67 percent
    average LoC: 3,424
    cost: 0.06 dollars
```

Step 2: related element localization.

The paper compares complete file input with a compressed skeleton format.

```text
complete file:
    contains ground truth: 53.67 percent
    average LoC: 778
    cost: 0.15 dollars

skeleton format:
    contains ground truth: 58.33 percent
    average LoC: 698
    cost: 0.02 dollars
```

The skeleton result is important. It shows that more file content can reduce
localization performance and raise cost. The model benefits from a structured
summary of classes, functions, and variables.

Step 3: edit-location localization.

```text
greedy:
    contains ground truth: 50.67 percent
    average LoC: 189
    cost: 0.06 dollars

direct from file-level:
    contains ground truth: 47.00 percent
    average LoC: 208
    cost: 0.18 dollars

multi-samples merged:
    contains ground truth: 56.33 percent
    average LoC: 342
    cost: 0.07 dollars

multi-samples as separate location sets:
    each set has roughly 48-50 percent ground truth containment
    average LoC roughly 165-213
    cost: 0.07 dollars
```

Agentless chooses separate sampled location sets instead of merging them because
merged contexts grow larger and can hurt repair. This is directly relevant to
evolution: diversity can be preserved as separate candidate families instead of
collapsed into one giant context.

Local microlib:

```text
localization_pipeline/
    file_localizer
    repo_structure_summarizer
    skeleton_builder
    related_element_localizer
    edit_location_sampler
    location_set_store
```

## Repair

Agentless builds a context window around each localized edit location. The paper
uses plus/minus 10 lines around each edit location. It samples:

```text
4 edit-location sets per issue
10 patches per edit-location set
40 total candidate patches per issue
```

The repair prompt asks for simple edit commands or search/replace diff edits,
not arbitrary repo manipulation. The implementation confirms this shape:
`repair.py` builds bounded file context and then parses generated edit blocks,
search/replace blocks, or diff commands into concrete patches.

Repair ablation from Table 3:

```text
greedy location, 40 samples:
    88 fixes
    29.33 percent
    cost: 0.22 dollars

multi-samples merged, 40 samples:
    85 fixes
    28.33 percent
    cost: 0.24 dollars

multi-samples separate, 4 x 10 samples:
    96 fixes
    32.00 percent
    cost: 0.29 dollars
```

Figure 6 reports that one greedy patch per location set already solves 80
issues, adding samples improves performance, and the curve plateaus around 40
samples under their selection method. It also reports an upper bound of 126
issues, or 42.0 percent, if all sampled patches could be considered rather than
selecting one final patch.

Local implications:

```text
sample diversity matters:
    multiple location sets can be better than one merged large context

selection is the bottleneck:
    if sampled candidates contain many potential successes, ranking needs work

patch format matters:
    small diff/search-replace edits are easier to parse and guard
```

For `codex exec`, this suggests a very simple first runner:

```text
for each target problem:
    build K context packets
    run codex exec K x M times with a strict patch schema
    parse each patch
    guard each patch
    run visible evaluator
    rank candidates
```

No autonomous loop required.

## Patch Validation

Agentless validates patches in three layers:

```text
majority voting:
    normalize patches
    select the most frequent candidate shape

regression tests:
    run existing tests that passed in the original repo
    ask LLM to remove tests likely to be non-regression tests
    keep patches with lowest regression failures

reproduction tests:
    generate tests from the issue
    keep tests that reproduce the issue on the original repo
    use them to filter patches that output issue resolved
    fall back to regression-only selection if no patch passes reproduction test
```

Patch validation ablation from Table 4:

```text
majority voting:
    77 fixes
    25.67 percent
    extra cost: 0.00 dollars

majority plus regression test:
    81 fixes
    27.00 percent
    extra cost: 0.01 dollars

majority plus regression plus reproduction test:
    96 fixes
    32.00 percent
    extra cost: 0.25 dollars
```

The generated reproduction tests are useful but noisy. Out of 300 SWE-bench Lite
problems, Agentless produces 213 tests that output the required issue-reproduced
message on the original repo. When the ground-truth patch is applied, only 94 of
those tests correctly output issue resolved.

This is the most important red-team point. Generated tests can improve patch
selection even when many are not fully plausible, but they are not trustworthy
final judges.

For DynaWorld:

```text
generated tests:
    candidate-visible ranking signal

repo-owned tests / hidden gates:
    promotion signal
```

The system should never promote a microlib only because a generated test passes.

## Main Results

Paper headline:

```text
SWE-bench Lite:
    96 / 300 fixed
    32.00 percent
    average cost: 0.70 dollars
    model: GPT-4o
```

The paper emphasizes that many higher leaderboard entries are closed-source or
commercial and do not release trajectories, while Agentless is open-source and
cheap. It reports Agentless as the highest-performing open-source approach on
SWE-bench Lite at the time of the paper.

SWE-bench Verified:

```text
Agentless with GPT-4o:
    194 / 500 fixed
    38.80 percent
```

The paper says Agentless performs best among techniques using GPT-4o on Verified
in their comparison. The current repository README reports later Claude 3.5
Sonnet integrations with higher numbers, but the paper's mechanism does not
depend on that.

## Benchmark Analysis

Agentless also studies SWE-bench Lite itself. This matters because the benchmark
defines what the solver is optimizing.

Manual categories:

```text
description quality:
    contains enough information
    contains reproducible example
    contains partial reproducible example
    does not contain enough information

solution in description:
    no solution
    partial steps
    complete steps
    exact patch
    misleading solution or steps

location information:
    exact line
    function
    file
    stack trace
    natural language
    keywords
    none
```

Important findings:

```text
10.0 percent:
    issue does not contain enough information

4.3 percent:
    exact ground-truth patch appears in issue description

9.7 percent:
    exact steps are described in natural language

5.0 percent:
    issue contains misleading solution or steps

less than 10 percent:
    exact lines are provided

about half:
    file-level location information is provided
```

They construct SWE-bench Lite-S by removing problems with exact patches,
misleading solutions, or insufficient information, leaving 249 problems.

Local transfer:
    Our first DynaWorld task suite should be sanitized the same way. A target is
    not a fair evolver benchmark if:

```text
it gives away the patch
it lacks enough information to solve
it has a misleading accepted solution
it has no stable evaluator
it rewards formatting instead of behavior
```

## Relationship To SWE-agent

SWE-agent says:

```text
interface design matters for autonomous trajectories
```

Agentless says:

```text
maybe do not use autonomous trajectories until a staged baseline fails
```

These are compatible. The local conclusion is:

```text
stage 0:
    one-shot Codex patch baseline

stage 1:
    Agentless-style localize/sample/validate/rank

stage 2:
    SWE-agent-style interactive repair for failures that need environment
    feedback and multi-step diagnosis

stage 3:
    AlphaEvolve-style population search over context packets, prompts, patches,
    tests, and evaluator variants
```

Do not skip stage 1. If Agentless-style sampling gets most of the win, the
evolver should focus on better localization, better generated tests, and better
rankers rather than building a complex action loop.

## What Transfers To `codex exec`

Agentless maps cleanly to batch Codex calls:

```text
localization call:
    codex exec "Given repo skeleton and task, return ranked files/elements..."

repair call:
    codex exec "Given issue and bounded snippets, return search/replace patch..."

test-generation call:
    codex exec "Given issue, write a reproduction test with fixed output markers..."

rerank call:
    codex exec "Given patches and logs, pick the most robust patch..."
```

But the outer Python runner should decide the stage order. The LLM should not
decide whether to localize, repair, test, or submit in the baseline.

Required artifacts per candidate:

```text
location_set_id
localized_files
localized_elements
edit_locations
context_window
patch_sample_id
parsed_patch
syntax_guard_result
regression_result
generated_repro_test_id
repro_result_original
repro_result_patched
normalized_patch_key
rank_score
promotion_result
```

This is more structured than a general agent transcript and easier to analyze.

## Microlibs Suggested By This Paper

```text
agentless_baseline_runner/
    Orchestrates localize -> sample patches -> validate -> rank.

repo_structure_summarizer/
    Builds a tree/skeleton of files, classes, functions, and symbols.

file_localizer/
    Combines prompt-based and embedding/BM25 retrieval.

related_element_localizer/
    Narrows files to classes/functions/global vars from skeleton context.

edit_location_sampler/
    Produces multiple small edit-location sets without merging all context.

context_window_builder/
    Builds plus/minus-N-line snippets around locations.

patch_sampler/
    Calls Codex multiple times with strict search/replace or unified-diff schema.

patch_parser/
    Turns model output into concrete diffs and rejects malformed output.

generated_repro_test_builder/
    Produces candidate tests with explicit issue-reproduced/issue-resolved
    marker contract.

regression_selector/
    Finds passing existing tests and excludes tests likely to be patched by the
    solution.

patch_ranker/
    Combines majority, regression, reproduction, syntax, and hidden scores.

benchmark_sanitizer/
    Flags target tasks that leak patch, lack enough info, or have ambiguous
    accepted behavior.
```

## Target Problems In This Repo

Agentless is a strong fit for local targets where localization is the hard part
but evaluation is cheap:

```text
anti-pattern cleanup:
    localize P1-P5 patterns
    sample minimal refactor patches
    rank by detector reduction plus smoke tests

config-schema repair:
    localize scattered cfg.get/default use
    sample patches that centralize normalization
    rank by unit smoke and LOC reduction

benchmark harness ergonomics:
    localize duplicated setup/serialization blocks
    sample helper extraction
    rank by test pass and line count

paper-note synthesis:
    weaker executable feedback
    use only for prompt/context experiments, not primary evolution benchmark

renderer kernels:
    too expensive/risky for first Agentless baseline
    only use after patch guard and hidden gates are reliable
```

The first runnable target could be:

```text
task:
    reduce one AGENTS.md P1/P2/P3 anti-pattern in a selected small file

visible evaluator:
    grep/detector count decreases
    py_compile or focused pytest passes

hidden evaluator:
    canonical smoke command
    diff scope check
    no unrelated file edits
```

This target would test the staged pipeline before expensive research objectives.

## Red-Team Notes

Risk: Agentless can overfit generated tests.
    Generated reproduction tests are noisy. Only 94 of 213 selected tests are
    plausible under the paper's ground-truth-patch check. Use generated tests as
    ranking signals, not final promotion gates.

Risk: Localization recall bounds the whole method.
    If the correct file/function is absent from the context packet, patch
    sampling is mostly wasted. Track oracle localization separately.

Risk: Majority voting suppresses rare correct patches.
    The paper's Figure 6 suggests the sample pool may contain more correct
    patches than the final selector can recover. A better local ranker could be
    more valuable than more samples.

Risk: Separate sampled contexts can duplicate cost.
    Separate location sets keep contexts small but may repeat work. Cache
    snippets, parser results, and evaluator logs aggressively.

Risk: Benchmark leakage.
    Agentless highlights exact patches and exact steps in benchmark issue text.
    Local tasks must be checked for leakage before claiming improvement.

Risk: Too much staging can miss exploratory fixes.
    Some problems require interactive diagnosis. Agentless should be baseline,
    not dogma. Escalate to SWE-agent/LATS-style loops only for failure classes
    that staged repair cannot solve.

## Local Falsification Tests

1. One-shot versus Agentless baseline:

```text
A: single codex exec patch call with task and relevant file
B: localize/sample/validate/rank with 4 x 5 patches

Measure:
    pass rate
    cost
    patch size
    unrelated edits
```

2. Separate versus merged context:

```text
A: merge all localized snippets into one prompt
B: run separate patch samples per location set

Measure:
    pass rate
    prompt length
    duplicated edits
    selection quality
```

3. Generated test value:

```text
A: majority only
B: majority plus regression
C: majority plus regression plus generated tests
D: generated tests plus hidden validation
```

Measure whether generated tests help rank without causing hidden failures.

4. Oracle localization:

```text
A: retrieved/localized context
B: oracle known file/function context
```

If oracle context works and retrieved context fails, improve localizer before
changing mutation/search.

5. Patch ranker upper bound:

```text
sample N patches
run hidden evaluator on all patches offline
compare:
    best possible in sample pool
    ranker-selected patch
```

This directly tests whether ranker or generation is the bottleneck.

## Design Consequences

Agentless changes the local priority order:

```text
before:
    build agent loop
    add memory
    add evolution

after:
    build task suite
    build deterministic staged baseline
    measure localization/sample/rank bottlenecks
    only then add autonomous loops where they solve measured failures
```

For the `alpha_evolve/` folder, this suggests the first implementation should
look like:

```text
alpha_evolve/
    tasks/
    context/
    localize/
    patch/
    validate/
    rank/
    archive/
```

not:

```text
alpha_evolve/
    big_agent.py
```

The staged layout still supports evolution later. Evolution can mutate prompts,
context builders, location sets, patch samples, generated tests, and ranker
weights. The difference is that every mutation has a measurable stage-level
effect.

## Open Questions For Later Papers

- Do OpenHands-style platforms add enough value beyond staged Agentless to
  justify their complexity for a local repo?
- How much of HumanEval/pass@k selection theory transfers to repo-level patch
  sampling?
- Can CodeT-style generated tests improve Agentless patch selection without
  increasing hidden-regression risk?
- What is the right first local ranker: majority, visible score, hidden score
  proxy, or model-based patch critique?
- Should AlphaEvolve evolve the patch code, the localizer, the generated tests,
  or the ranker first?
