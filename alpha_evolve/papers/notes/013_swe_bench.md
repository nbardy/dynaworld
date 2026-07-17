# 013 - SWE-bench: Can Language Models Resolve Real-World GitHub Issues?

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2310.06770
    https://arxiv.org/pdf/2310.06770
    https://github.com/SWE-bench/SWE-bench
    https://www.swebench.com/original.html
    https://swebench.com/SWE-bench/

Implementation artifacts inspected:
    https://github.com/SWE-bench/SWE-bench/blob/main/README.md
    https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/run_evaluation.py
    https://github.com/SWE-bench/SWE-bench/blob/main/swebench/harness/grading.py
    https://github.com/SWE-bench/SWE-bench/blob/main/docs/guides/evaluation.md

Bibliographic metadata:
    Authors: Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao,
    Kexin Pei, Ofir Press, Karthik Narasimhan.
    First arXiv submission: 2023-10-10.
    Latest arXiv version inspected: v3, 2024-11-11.
    Venue/context: ICLR 2024 oral.

Why this paper matters for alpha_evolve:
    SWE-bench is not an evolution algorithm. It is the benchmark that forces
    agentic code systems to stop pretending that toy function synthesis is
    repo-level software engineering. For `alpha_evolve`, it defines what a real
    local target should look like:

```text
issue / problem statement
base commit or reproducible codebase state
patch prediction
containerized evaluation
fail-to-pass tests
pass-to-pass maintenance tests
logs and report
```

    The biggest transfer is evaluator design. A Codex evolver needs its own
    SWE-bench-like microbenchmarks for DynaWorld: replayable tasks, real tests,
    maintenance checks, and patch-level scoring. Without that, AlphaEvolve-style
    search will optimize whatever shallow smoke test we accidentally expose.

One-sentence mechanism:
    Build task instances from real GitHub issue/pull-request pairs, apply model
    patches to the repository at the PR base commit, run tests contributed by
    the PR before/after the gold patch to identify fail-to-pass cases, then
    grade predicted patches by whether they make fail-to-pass tests pass while
    preserving pass-to-pass tests.

## Reading Questions

- What is the executable feedback signal?
  Repository tests. The core signal is fail-to-pass tests: tests that fail before
  the reference PR solution and pass after it. Maintenance is checked with
  pass-to-pass tests that should remain passing after the model patch.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  SWE-bench itself does not search. It evaluates patch predictions. For an
  alpha_evolve runner, each candidate is a patch applied to a reproducible repo
  state.

- What is the population/database/selection mechanism?
  None in the benchmark. The harness accepts predictions, applies patches, runs
  tests, parses logs, and reports resolved/not resolved. Selection is outside
  the paper. That is exactly why it is useful as a final judge for agents,
  search, or evolution.

- What evidence proves the benchmark is hard?
  The paper reports that with BM25 retrieval, Claude 2 resolves 1.96 percent of
  the original SWE-bench issues. Even oracle retrieval only raises Claude 2 to
  4.8 percent in the paper's analysis. Current repo docs and the benchmark site
  have evolved after the paper, but the paper's core finding stands: real issue
  repair is dominated by localization, context management, patch format,
  multi-file reasoning, and test preservation.

- What does the method assume that DynaWorld does not have?
  SWE-bench assumes issue/PR/test-patch pairs already exist. DynaWorld local
  alpha_evolve targets may not have natural GitHub issues and PR test patches.
  We need to manufacture small replayable tasks with the same structure.

## Benchmark Construction

The paper collects task instances from real open-source Python repositories.

Pipeline:

```text
Stage I:
    scrape pull requests from 12 popular Python repositories

Stage II:
    keep merged PRs that resolve an issue and contribute tests

Stage III:
    run execution filtering
    apply test patch
    run tests before gold patch
    apply gold patch
    run tests after gold patch
    keep instance only if at least one test changes fail -> pass
```

Scale:

```text
raw PRs:
    about 90,000

final SWE-bench instances:
    2,294

repositories:
    12 popular Python repos

Lite subset:
    300 instances, sampled to be more self-contained and bug-fix focused

training set:
    19,000 issue/PR pairs from 37 additional repositories for SWE-Llama
```

The task instance has four conceptual parts:

```text
C:
    codebase at base commit

P:
    problem statement from issue title/body/comments before PR initial commit

T:
    test patch / tests used for evaluation

delta:
    gold patch from the PR's non-test code changes
```

Important filtering:

- require install success;
- require PR passes tests after the gold patch;
- require at least one fail-to-pass test;
- exclude tasks whose tests invoke newly created functions/classes first
  introduced by the solution, because the name may be impossible to infer from
  the issue.

This is exactly the kind of filtering a local DynaWorld task suite needs. Do not
create tasks whose hidden test depends on a new API name the agent cannot know.

## Task Formulation

Input to model:

```text
issue text
codebase or retrieved code context
instructions
example patch format
```

Output from model:

```text
patch file / unified diff
```

Evaluation:

```text
apply patch
run repository tests
parse test logs
resolved = all fail-to-pass pass and all pass-to-pass remain pass
```

The current harness implements patch application with several fallback commands:

```text
git apply --verbose
git apply --verbose --reject
patch --batch --fuzz=5 -p1 -i
```

Then it runs an eval script in Docker and writes logs/report files. The grading
code computes:

```text
fail-to-pass:
    did the issue-specific failing tests pass?

pass-to-pass:
    did already-passing tests stay passing?

resolved:
    fail-to-pass == 1 and pass-to-pass == 1
```

This is a stronger contract than "the candidate improved the metric." It tests
both repair and regression.

## Dataset Characteristics

From the paper's main statistics:

```text
issue text:
    mean 195.1 words
    max 4477 words

codebase:
    mean 3,010 non-test files
    mean 438K non-test lines

gold patch:
    mean 32.8 edited lines
    mean 1.7 files edited
    mean 3 functions edited

tests:
    mean 9.1 fail-to-pass tests
    mean 120.8 total tests
```

The repo-level scale matters. Most evolution papers earlier in this queue search
over small functions or heuristics. SWE-bench says real patch tasks often need
multi-file context and regression preservation.

For DynaWorld, target tasks should include:

```text
one-file easy repairs
multi-file contract repairs
config/data/renderer integration repairs
performance changes with regression tests
documentation/schema updates with tests
```

If the target suite is only single-file toy tasks, it will overstate the value of
the evolver.

## Retrieval And Context

SWE-bench baselines cannot fit whole repos in context, so they use BM25 file
retrieval and an oracle retrieval analysis.

Key paper observations:

```text
BM25 at 27k token context:
    retrieves all oracle files in about 39.83 percent of instances
    retrieves any oracle file in about 51.27 percent
    retrieves none of the oracle files in almost half
```

Increasing context can hurt:

```text
Claude 2 BM25:
    13k context -> 1.96 resolved
    27k context -> 1.87
    50k context -> 1.22
```

This is one of the most important local lessons. More context is not
automatically better. For `codex exec`, dumping the entire DynaWorld tree into a
prompt is not a strategy. The runner needs retrieval and localization as a
first-class stage.

Oracle-collapsed context improves performance:

```text
only oracle files, collapsed to edited lines +/- 15
GPT-4: 1.3 -> 3.4
Claude 2: 4.8 -> 5.9
```

Interpretation:

```text
localization and context compression are part of the task
```

Local microlibs should include issue-aware retrieval and code-context pruning
before candidate evolution.

## Results And Failure Analysis

BM25 main table in the v3 paper:

```text
SWE-bench:
    Claude 3 Opus: 3.79 resolved
    Claude 2: 1.97
    ChatGPT-3.5: 0.17
    GPT-4-turbo: 1.31
    SWE-Llama 7b: 0.70
    SWE-Llama 13b: 0.70

SWE-bench Lite:
    Claude 3 Opus: 4.33
    Claude 2: 3.00
    ChatGPT-3.5: 0.33
    GPT-4-turbo: 2.67
    SWE-Llama 7b: 1.33
    SWE-Llama 13b: 1.00
```

The paper's earlier headline says Claude 2 resolves 1.96 percent with BM25.
The current v3 table includes Claude 3 Opus and reports Claude 2 at 1.97.

Patch application is itself hard:

```text
models often fail to emit applicable patches
applied patch rate is far higher than resolved rate
```

Models under-edit:

```text
all gold patches average:
    74.5 total edited lines
    1.7 files
    3.0 functions

model generated applied patches:
    much shorter
    often one file
    simpler and greedier
```

Qualitative finding:

```text
models often edit the right function but miss surrounding config/style/logical
constraints
```

The Sphinx example in the paper shows the model changed the right function but
failed to respect the `napoleon_use_param` configuration branch, so tests for
the opposite config failed.

Local implication:

```text
candidate-visible gates must include regression/maintenance tests
```

If a candidate only passes the newly failing test but breaks adjacent behavior,
it is not solved.

## Harness Lessons

### Patch Format Is A Contract

SWE-bench evaluates patches, not prose. The harness expects:

```json
{
  "instance_id": "...",
  "model_name_or_path": "...",
  "model_patch": "diff --git ..."
}
```

For `alpha_evolve`, candidate output should also be structured:

```text
candidate patch diff
metadata JSON
evaluation artifacts
```

Avoid a free-form "here is my solution" format.

### Docker Or Equivalent Isolation Is Required

Current SWE-bench uses Docker for reproducible evaluation. The README warns that
evaluation can require substantial disk, RAM, and CPU. The exact resource
numbers are benchmark-specific, but the design principle is general:

```text
the evaluator owns the environment
candidate patches do not run in the user's dirty worktree
```

For DynaWorld, full Docker may be expensive or awkward for Metal/GPU work, but
we still need isolation:

```text
git worktree
venv fingerprint
config fingerprint
artifact directory
timeout
cleanup
```

### Resolved Means Fix Plus Maintenance

The grading code:

```text
resolved = fail-to-pass all pass and pass-to-pass all pass
partial = some fail-to-pass pass and pass-to-pass pass
no = otherwise
```

Local evolution should copy this exact split:

```text
repair score:
    target failures fixed

maintenance score:
    previous behavior preserved

promotion:
    both are true
```

This matters for performance or renderer work, where a patch can improve one
case by breaking another.

## Design Implications For `alpha_evolve`

### Build DynaWorld-SWE Microbenchmarks

Each target problem should be packaged as:

```text
problem_id
base_ref
problem_statement
allowed_files
visible_tests
hidden_fail_to_pass
hidden_pass_to_pass
setup_command
eval_command
timeout
expected_artifacts
```

The base_ref may be a git commit or a patch bundle. The point is replayability.

### Treat Localization As A Stage

SWE-bench shows that retrieval failure dominates. Add:

```text
context_retriever
oracle_context_simulator for local ablations
context_pruner
localization_score
```

Before evolving patches, run localization baselines:

```text
known files / oracle files
rg/BM25 retrieved files
Codex-selected files
hybrid retrieved + dependency graph
```

If patch search only works with oracle files, the real bottleneck is retrieval.

### Candidate Output Should Be Patch-First

The paper finds whole-file regeneration performs worse than patch generation in
their setting. Local `codex exec` mutation should produce patch-like edits and
metadata, not full file rewrites, unless the target explicitly is a generated
file.

### Include Under-Edit Diagnostics

Model patches tend to be shorter and greedier than gold patches. Track:

```text
lines_added
lines_removed
files_touched
functions_touched
gold/expected scope if known
regression failures
```

This does not mean bigger patches are better. It means the runner should detect
when all candidates keep making one-line patches against a multi-file contract.

### Separate Visible And Hidden Tests

Use SWE-bench's fail-to-pass/pass-to-pass split:

```text
visible:
    cheap tests for candidate repair and reward

hidden:
    final fail-to-pass/pass-to-pass tests for promotion
```

Never let generated tests become the hidden evaluator.

## Proposed Microlibs

### `repo_task_schema`

Responsibility:

```text
define a DynaWorld issue-style task instance
```

Fields:

```text
problem_id
base_ref
problem_statement
allowed_paths
setup_command
visible_eval_command
hidden_eval_command
fail_to_pass
pass_to_pass
timeout_seconds
artifact_expectations
```

### `patch_prediction_format`

Responsibility:

```text
standardize candidate outputs as patch + metadata
```

Fields:

```text
candidate_id
parent_id
model_command
patch
summary
files_touched
declared_risks
```

### `fail_pass_grader`

Responsibility:

```text
compute repair and maintenance from test logs
```

Outputs:

```text
fail_to_pass_success
fail_to_pass_failure
pass_to_pass_success
pass_to_pass_failure
resolved
partial
```

### `context_retriever`

Responsibility:

```text
retrieve and compress repo context for an issue-style problem
```

Modes:

```text
rg lexical
BM25
dependency/neighborhood
oracle for ablation only
human-pinned
```

### `evaluation_sandbox`

Responsibility:

```text
apply patch, run eval, parse logs, cleanup
```

Must record:

```text
base_ref
patch_apply_method
env fingerprint
command
timeout
test log
diff before/after
```

### `task_instance_builder`

Responsibility:

```text
turn local TODOs/regressions into SWE-style task instances
```

Use it for:

```text
alpha_evolve microlib tasks
renderer regression tasks
config normalization tasks
data contract tasks
```

## Local Falsification Tests

### Test 1: Hidden Pass-To-Pass Catches Greedy Fixes

Create a local task where a one-line patch fixes the target test but breaks an
adjacent behavior.

Expected:
    fail-to-pass passes, pass-to-pass fails, candidate not promoted.

### Test 2: Retrieval Ablation Identifies Bottleneck

Compare:

```text
oracle file context
rg/BM25 context
Codex-selected context
full noisy context
```

Expected:
    If oracle works and retrieved fails, improve retrieval before mutation.

### Test 3: Patch Apply Fail Is A First-Class Failure

Feed malformed patches from candidate generation.

Expected:
    evaluator reports patch_apply_failed separately from test_failed.

### Test 4: Under-Edit Detector

Run several candidates on a multi-file contract task.

Expected:
    if all candidates touch only one shallow file while pass-to-pass fails,
    runner flags possible under-edit/local-minimum.

### Test 5: Replay From Base Ref

Run the same candidate twice from the same `base_ref`.

Expected:
    identical test status, or environment nondeterminism is reported.

## How It Changes The AlphaEvolve Plan

SWE-bench inserts a benchmark/harness layer under every search algorithm:

```text
repo_task_schema
context_retriever
evaluation_sandbox
fail_pass_grader
patch_prediction_format
```

Only after those exist should we compare:

```text
one-shot codex exec
Reflexion repair
ToT plan search
LATS candidate search
LLaMEA serial evolution
AlphaEvolve database/islands
```

Without a SWE-bench-like harness, search results are not meaningful.

## Target Problems In This Repo

Good SWE-style local targets:

```text
small alpha_evolve runner bugs
config normalization regressions
data contract invariants
renderer dispatch F=3 vs F!=3 behavior
prompt/schema serialization bugs
```

Bad first targets:

```text
long GPU model-quality improvements
open-ended research hypotheses without deterministic tests
manual-only visual quality judgments
```

For research targets, create a narrower SWE-style proxy first, then keep the
research metric as a separate acceptance layer.

## Open Questions

- What is the smallest local SWE-style task set that predicts useful DynaWorld
  changes?
- Should hidden tests be completely hidden from Codex prompts or just withheld
  from automatic reflection?
- How do we handle tasks where the "gold" solution is not known because the
  target is a new research improvement?
- Can we build oracle-context and retrieved-context splits for each local task?
- How often should task instances be refreshed to avoid overfitting the
  evolver?

## Bottom Line

SWE-bench is the evaluator realism check for this entire project. AlphaEvolve,
LATS, Reflexion, and Self-Refine are only useful if they are measured against
replayable repo tasks with fail-to-pass and pass-to-pass gates. The immediate
DynaWorld move is not another search algorithm. It is a SWE-bench-like local
harness for small, deterministic microlib and repo-contract tasks.
