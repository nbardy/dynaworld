# 014 - SWE-agent: Agent-Computer Interfaces Enable Automated Software Engineering

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2405.15793
    https://arxiv.org/pdf/2405.15793
    https://github.com/SWE-agent/SWE-agent
    https://swe-agent.com/latest/
    https://papers.nips.cc/paper_files/paper/2024/hash/5a7c947568c1b1328ccc5230172e1e7c-Abstract-Conference.html

Implementation artifacts inspected:
    https://github.com/SWE-agent/SWE-agent/blob/main/README.md
    https://github.com/SWE-agent/SWE-agent/blob/main/docs/config/config.md
    https://github.com/SWE-agent/mini-swe-agent

Bibliographic metadata:
    Authors: John Yang, Carlos E. Jimenez, Alexander Wettig, Kilian Lieret,
    Shunyu Yao, Karthik R. Narasimhan, Ofir Press.
    First arXiv submission: 2024-05-24.
    Venue/context: NeurIPS 2024.

Current implementation note:
    The main SWE-agent repository now says most current development effort has
    moved to mini-SWE-agent, which it describes as simpler while matching
    SWE-agent performance. This note is about the paper, but before copying
    runtime architecture we should inspect mini-SWE-agent too.

Why this paper matters for alpha_evolve:
    SWE-agent is a direct warning against giving an LLM raw shell power and
    expecting the model to discover a clean software-engineering workflow. The
    paper's point is that the agent-computer interface is an algorithmic object:
    command set, edit representation, observation shape, history compression,
    command validation, and feedback templates all change downstream solve rate.

    For the local `alpha_evolve/` plan, the Codex process should not only be:

```text
codex exec "<prompt>"
```

    It should be:

```text
target task
    -> curated context packet
    -> constrained prompt/tool contract
    -> patch or microlib edit
    -> explicit observation/evaluator report
    -> archived trajectory
    -> next prompt generated from structured feedback
```

    The important transfer is not the exact SWE-agent command names. The
    important transfer is that every interaction surface should be designed for
    the model's strengths and weaknesses, then measured as part of the search
    system.

One-sentence mechanism:
    Build an LM-friendly interface over the shell and file system, where the
    agent alternates thought and command, receives concise feedback from
    specialized search/view/edit commands, benefits from edit guardrails and
    history compression, then submits the resulting patch for SWE-bench-style
    evaluation.

## Reading Questions

- What is the executable feedback signal?
  SWE-agent receives command feedback during the trajectory and final
  execution-based benchmark feedback after submission. During a run, it can see
  search results, file snippets, edit results, linter errors, Python or pytest
  output, and explicit no-output confirmations. Final grading is still the
  SWE-bench resolved metric or HumanEvalFix pass@1.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  SWE-agent searches over an interactive trajectory: command sequences that
  inspect files, reproduce bugs, edit code, run checks, and submit a final
  patch. It is not population-based evolution, but its trajectories are exactly
  the unit that an AlphaEvolve-style outer loop could mutate, replay, and score.

- What is the population/database/selection mechanism?
  The paper has no evolutionary database. It runs one trajectory per task and
  reports pass@1. For local use, the missing layer is a trajectory/candidate
  database that stores prompt, context packet, commands, patch, visible
  evaluator results, hidden evaluator results, cost, and failure labels.

- What evidence matters?
  GPT-4 Turbo SWE-agent resolves 12.47 percent of the 2,294 full SWE-bench test
  instances and 18.00 percent of SWE-bench Lite. The Shell-only GPT-4 Turbo
  baseline resolves 11.00 percent on Lite. The earlier non-interactive RAG
  GPT-4 Turbo baseline resolves 1.31 percent full and 2.67 percent Lite. The
  paper therefore gives concrete evidence that interface design can be worth
  more than prompt wording alone.

- What does this assume that DynaWorld does not yet have?
  SWE-agent assumes a benchmark task can be expressed as an issue statement
  against a checked-out repository, with patch submission as the final action.
  DynaWorld alpha-evolve targets may include benchmark harness tuning, renderer
  kernels, trainer cleanup, and research-note generation. Those need their own
  task envelopes and promotion gates before the SWE-agent loop transfers cleanly.

## Mechanism

SWE-agent is an LM plus an ACI. The ACI includes:

```text
command set:
    special search, file viewer, edit, create, submit

observation format:
    line-numbered snippets
    concise search summaries
    explicit command outcome messages
    linter feedback
    collapsed older observations

history policy:
    system prompt
    optional demonstration
    issue statement
    recent observations kept in full
    older observations collapsed

execution environment:
    shell process in a sandboxed repo
    standard Linux commands allowed
    special commands layered above shell

final artifact:
    patch generated from all edits
```

The paper frames the ACI as the equivalent of an IDE for LM agents. Human UIs
are not automatically model-friendly. GUI affordances, rich screens, and raw
terminal streams can be wasteful or ambiguous for an LM. A good ACI should make
the relevant state legible, make high-value actions cheap, and prevent common
errors from cascading.

The ACI design principles are:

```text
simple actions:
    few options
    short documentation
    easy to learn in-context

compact actions:
    important operations should happen in one step
    avoid multi-turn rituals for common edit/navigation operations

concise feedback:
    enough state to understand the effect of the last action
    not enough output to flood or distract the context window

guardrails:
    catch malformed commands
    catch bad edits when possible
    tell the agent what failed and ask it to recover
```

For `alpha_evolve`, these principles imply that the runner interface is not a
secondary engineering detail. It is part of the optimizer. A weak interface can
make a strong model waste budget on navigation and broken edit mechanics.

## SWE-agent Commands

The paper's special commands include:

```text
open <path> [<line_number>]:
    open a file in the file viewer

goto <line_number>:
    move the current file window to a line

scroll_down / scroll_up:
    move the current file window

search_file <search_term> [<file>]:
    search inside the current or specified file

search_dir <search_term> [<dir>]:
    search files under a directory

find_file <file_name> [<dir>]:
    find files by name

edit <n>:<m> ... end_of_edit:
    replace an inclusive line range in the open file

create <filename>:
    create and open a new file

submit:
    generate the final patch and close the shell
```

The exact command surface is less relevant locally than the command semantics:

```text
view:
    bounded file window
    line numbers
    current file identity
    omitted line counts

search:
    summarized results
    cap on huge result sets
    nudge toward more specific queries

edit:
    one compact multi-line replacement
    immediate post-edit view
    syntax/lint rejection for obvious bad edits

submit:
    one final patch artifact
```

Codex already has a richer general-purpose interface than the paper's agent,
but a batch `codex exec` evolver does not automatically expose the right
surface. If a candidate prompt asks Codex to edit files directly in a full
workspace, the outer loop may only see a final diff and test log. The ACI lesson
is to make intermediate observations explicit in the prompt contract and the
trajectory archive.

## Search And Navigation

SWE-agent introduces `find_file`, `search_file`, and `search_dir` to avoid raw
shell search output. Search commands summarize results and cap them. If a query
has too many matches, the interface withholds the full dump and tells the agent
to narrow the query.

This matters because raw repo search is easy to misuse:

```text
grep/rg dump too large:
    context window fills with irrelevant matches

interactive next/previous search:
    model exhaustively iterates through matches
    cost budget disappears

no search abstraction:
    model spends turns inventing shell commands
```

The paper's ablation found that a summarized search interface was better than
iterative search. Iterative search resembled familiar human search UI, but
models often inspected every match one by one. Interfaces that are pleasant for
people may be bad for agents because agents do not naturally ignore optional
distractions.

For `alpha_evolve`, search should be split into two stages:

```text
retrieval stage:
    deterministic repo/context retriever builds the context packet

candidate stage:
    Codex can still ask for local search, but the outer loop captures queries,
    result sizes, and whether the candidate changed target files
```

If a candidate fails because it never inspected the right file, the failure is
not evidence against the mutation idea. It is a localization/interface failure.

## File Viewer

The file viewer shows:

```text
file path
total line count
window of file content
line numbers
number of omitted lines above and below
```

The default window in the final paper setup is 100 lines. The ablation compares
30 lines, 100 lines, and full-file view:

```text
30 lines:
    14.3 percent on SWE-bench Lite

100 lines:
    18.0 percent on SWE-bench Lite

full file:
    12.7 percent on SWE-bench Lite
```

This is a clean example of context shaping. Too little context loses local
structure. Full files add distracting stale or irrelevant text. The model wants
bounded context with enough local syntax and line numbers to make precise edits.

Local implication:

```text
alpha_evolve/context_packet.py
    should create bounded file windows
    should include stable line anchors or symbol anchors
    should avoid dumping whole trainers unless the task demands it
```

For DynaWorld, this is especially important because trainer files are long and
contain many adjacent experiments. Whole-file context invites accidental
refactors and unrelated edits.

## File Editor

The edit command is the paper's highest-value interface design. It consolidates
the common edit workflow into one action:

```text
choose open file
choose line range
provide replacement text
apply patch
show updated window
reject obvious syntax/lint failures
```

The ablation table:

```text
edit with linting:
    18.0 percent on SWE-bench Lite

edit without linting:
    15.0 percent

no edit command:
    10.3 percent
```

The paper's analysis says editing remains hard even with the custom interface:
51.7 percent of GPT-4 Turbo trajectories on full SWE-bench have at least one
failed edit. An edit attempt has a 90.5 percent chance of eventually being
successful, but after one failed edit the recovery probability drops to 57.2
percent.

Local implication:
    The `codex exec` outer loop should treat edit failures as a distinct failure
    class. A candidate that has a good idea but repeatedly produces malformed
    patches should be salvageable by an edit-repair lane. A candidate that edits
    the wrong function needs localization repair. A candidate that passes
    syntax but fails behavior needs semantic mutation.

Potential microlib:

```text
patch_guard.py
    apply generated patch to temp worktree
    run syntax/import checks
    reject patch before expensive evaluator
    capture exact rejected hunk and error

patch_repair_prompt.py
    feed failed patch plus guardrail error to Codex
    request minimal corrected patch
    forbid re-running same failed patch
```

This is not the same as final fitness. It is pre-evaluator hygiene.

## Context Management

SWE-agent manages context by:

```text
system prompt:
    describes environment and commands

demonstration:
    optional successful trajectory

instance prompt:
    issue statement and operating advice

turn format:
    discussion/thought plus a single command

environment response:
    command output or explicit no-output message

history compression:
    recent observations kept
    older observations collapsed to one-line placeholders
```

The paper keeps the last five observations in full in the default configuration.
The ablation table:

```text
last 5 observations:
    18.0 percent on SWE-bench Lite

full history:
    15.0 percent

without demonstration:
    16.3 percent
```

Important transfer:
    More trajectory context is not automatically better. Old observations may
    contain stale file views or command output that no longer reflects the
    current repo state. For local evolution, we should store full history in the
    database, but feed only a curated recent/history summary back into the next
    Codex prompt.

Proposed local split:

```text
archive:
    full transcript
    every command
    every stdout/stderr
    every patch
    every evaluator log

prompt context:
    task statement
    current candidate summary
    recent relevant observations
    selected failure reflection
    current file windows
    evaluator report digest
```

The archive is for audit and future mining. The prompt context is for model
performance.

## Evaluation Setup And Results

Datasets:

```text
SWE-bench full:
    2,294 task instances
    12 Python repositories
    real GitHub issue/PR-derived repairs

SWE-bench Lite:
    300 self-contained functional bug fixes

HumanEvalFix:
    short-form code debugging benchmark
```

Models:

```text
GPT-4 Turbo:
    gpt-4-1106-preview
    128k context

Claude 3 Opus:
    claude-3-opus-20240229
    200k context
```

Main SWE-bench results from Table 1:

```text
RAG with GPT-4 Turbo:
    full: 1.31 percent resolved
    Lite: 2.67 percent resolved

RAG with Claude 3 Opus:
    full: 3.79 percent resolved
    Lite: 4.33 percent resolved

Shell-only GPT-4 Turbo:
    Lite: 11.00 percent resolved

Shell-only without demonstration:
    Lite: 7.33 percent resolved

SWE-agent with GPT-4 Turbo:
    full: 12.47 percent resolved
    Lite: 18.00 percent resolved

SWE-agent with Claude 3 Opus:
    full: 10.46 percent resolved
    Lite: 13.00 percent resolved
```

HumanEvalFix:

```text
SWE-agent with GPT-4 Turbo:
    Python: 87.7 pass@1
    JavaScript: 89.7 pass@1
    Java: 87.9 pass@1
```

Interpretation:
    The result is not that SWE-agent solves software engineering. It solves a
    much larger fraction than non-interactive baselines from the same era, and
    it shows that changing the interface can yield a major gain without changing
    model weights.

## Ablations

Paper Table 3 gives the clearest design guidance:

```text
Editor:
    edit with linting: 18.0
    edit without linting: 15.0
    no edit: 10.3

Search:
    summarized search: 18.0
    iterative search: 12.0
    no search: 15.7

File viewer:
    30 lines: 14.3
    100 lines: 18.0
    full file: 12.7

Context:
    last 5 observations: 18.0
    full history: 15.0
    without demonstration: 16.3
```

Key lessons:

```text
compact multi-line edits beat raw shell editing
linting/guardrails matter
summarized search beats human-like iterative search
bounded file windows beat full files
compressed history beats full transcript in prompt
demonstrations help but are smaller than the edit/search/view effects
```

For a Codex evolver, these are ablation targets. We can run local A/B tests:

```text
raw prompt with whole files
bounded context packet
bounded context plus evaluator digest
bounded context plus reflection
bounded context plus explicit patch schema
bounded context plus edit-failure repair
```

The system should not assume the most expressive interface is best. It should
measure interface variants as part of the population.

## Trajectory Patterns

The paper finds recurring phases:

```text
phase 1:
    reproduce or localize

phase 2:
    zoom from broad search to specific file/line

phase 3:
    edit and execute loop

phase 4:
    submit or exit by budget
```

Resolved instances tend to finish earlier and cheaper. The paper reports that
successful GPT-4 Turbo runs have median cost 1.21 dollars and 12 steps, while
unsuccessful runs average 2.52 dollars and 21 steps. 93.0 percent of resolved
instances are submitted before exhausting the cost budget, compared with 69.0
percent overall.

Local implication:
    A long-running candidate trajectory should not automatically receive more
    budget. In the SWE-agent setting, slow failure is common. For `alpha_evolve`,
    we need a budget controller that detects when a candidate is looping:

```text
signals:
    repeated same command
    repeated failed patch class
    many edits without visible score improvement
    localization churn after evaluator already identifies failing area
    rising cost without new information

actions:
    stop candidate
    summarize failure
    generate reflection
    spawn a new candidate from a different parent/context
```

This maps cleanly to an evolutionary loop: kill trajectories that fail slowly;
preserve their useful failure notes.

## Failure Modes

The paper uses GPT-4o to categorize unresolved SWE-agent trajectories on
SWE-bench Lite, with author agreement of 87 percent on a hand-labeled validation
set. The biggest buckets:

```text
incorrect implementation:
    model edits but does not solve the issue

overly specific implementation:
    model solves too narrow a case

failed to recover from edit:
    malformed/bad edits cascade

failed to find edit location:
    correct file/line not localized

failed to find relevant file:
    broader localization failure

gave up prematurely:
    stopped despite not solving

cannot reproduce:
    no useful visible bug signal

ran out of time:
    budget exhaustion
```

The paper states that about half of unresolved instances are incorrect or overly
specific implementations, and cascading failed edits are another large failure
class.

For `alpha_evolve`, these categories should be first-class failure labels in the
candidate database. They tell us what mutation operator to use next:

```text
incorrect implementation:
    semantic mutation
    stronger evaluator feedback

overly specific implementation:
    broaden tests
    hidden-case prompt
    generated counterexamples

failed edit:
    patch repair
    smaller hunk
    syntax guard

failed localization:
    new context retrieval
    symbol/trace guided localization

cannot reproduce:
    build reproducibility harness
    isolate deterministic seed/test

ran out of time:
    stop expanding this branch
```

This is more useful than a generic failed score.

## What Transfers To `codex exec`

The user note asks specifically for evolve using Codex prompt execution. Current
Codex CLI usage is:

```text
codex exec "<prompt>"
```

`codex -p` is a profile flag, not the prompt-execution form. The local runner
should still preserve the user's phrase "codex -p" in planning notes as the
intended Codex prompt-call mechanism, but implementation should call the actual
CLI form after verifying installed Codex version.

The SWE-agent translation:

```text
SWE-agent command:
    open / search / edit / run / submit

Codex evolver outer-loop equivalent:
    build context packet
    call codex exec with a strict role and output schema
    require patch or file edits in a candidate worktree
    diff the worktree
    run guard checks
    run visible evaluator
    optionally run hidden evaluator
    archive transcript, diff, scores, logs, and failure label
```

Codex itself can use tools interactively, but the evolutionary runner should not
depend on unstructured interaction. Every candidate should end in a machine
readable outcome:

```text
candidate_id
parent_id
prompt_id
context_packet_id
codex_command
stdout_log
stderr_log
changed_files
patch_path
visible_scores
hidden_scores
guardrail_failures
failure_label
promotion_decision
```

The ACI lesson is that the prompt and filesystem arrangement given to Codex are
the interface. We can make that interface agent-friendly:

```text
give Codex a small task root
include a concise problem file
include bounded relevant snippets
include evaluator command(s)
include a strict patch scope
include a final-report schema
keep unrelated repo state out of candidate context
```

## Microlibs Suggested By This Paper

```text
aci_contract/
    Defines the model-facing task packet:
        instructions
        allowed file scope
        evaluator commands
        output schema
        patch constraints

context_packet/
    Builds bounded, line-numbered context snippets from repo files.
    Tracks source file, line window, symbol, and retriever reason.

search_summary/
    Wraps rg/find results into capped, stable summaries.
    Records overflow and asks for narrower retrieval instead of dumping all.

patch_guard/
    Applies candidate diffs in a temp worktree.
    Runs syntax/import/format checks before expensive evaluators.
    Rejects obvious bad patches with structured error reports.

observation_digest/
    Converts command/evaluator output into concise observations.
    Keeps full logs in archive but emits prompt-safe summaries.

history_compactor/
    Keeps recent observations verbatim and older observations as one-line
    summaries with log pointers.

trajectory_store/
    Stores prompt, Codex command, transcript, patch, changed files, evaluator
    logs, cost, score, and failure labels.

budget_controller/
    Stops candidates that loop, exceed cost, or fail to produce new information.

failure_classifier/
    Labels failed candidates as localization, edit, semantic, overly-specific,
    reproducibility, evaluator, or timeout failures.

interface_ablation_runner/
    Compares prompt/context/interface variants on the same task suite.
```

These are intentionally small. The local AlphaEvolve loop should not start with
a monolithic SWE-agent clone. It should start with reusable microlibs around
Codex batch calls.

## Target Problems In This Repo

Good first targets for this ACI layer:

```text
config normalization cleanup:
    constrained file scope
    deterministic tests
    easy LOC/reduction metrics
    failure modes visible in smoke tests

anti-pattern detectors:
    evaluator can measure P1-P5 instances
    candidates can propose focused refactors
    hidden smoke prevents semantic damage

renderer benchmark harness cleanup:
    visible evaluator can run fast benchmark/verification scripts
    hidden evaluator can run stricter artifacts only for promoted candidates

paper/agent-note synthesis:
    weaker executable signal
    useful for prompt/interface experiments but not first fitness target

PowerFoam verifier routing:
    high stakes and expensive
    only after runner has guardrails and budget controls
```

The best first SWE-agent-inspired task suite is probably not a renderer kernel.
It is a set of small repo-cleanup or harness tasks where:

```text
task statement is crisp
file scope is small
evaluator is cheap
hidden gate exists
desired behavior is objective
```

This will test whether the ACI around `codex exec` improves candidate quality
before we spend search budget on expensive training or rendering objectives.

## Red-Team Notes

Risk: Overfitting to interface ceremony.
    A polished ACI can make agent transcripts look more competent without
    improving final patches. Always judge final patches with evaluator gates.

Risk: Hiding too much context.
    Search summaries and bounded file windows can hide the actual relevant
    code. Track oracle-context ablations. If oracle context solves the task but
    retrieved context fails, do not blame Codex mutation.

Risk: Guardrails becoming the objective.
    Syntax/lint guards are useful, but a syntactically clean patch can be a bad
    patch. Guardrails should only filter obvious waste before the real
    evaluator.

Risk: Candidate-visible tests induce narrow fixes.
    SWE-agent has an overly-specific implementation failure class. DynaWorld
    needs hidden/pass-to-pass gates and generated counterexamples to prevent
    candidates from fitting only the visible smoke.

Risk: Full transcript in prompt harms performance.
    The database should store all logs, but prompts should receive compressed
    state. Full logs are for audit, not necessarily for next-candidate context.

Risk: Slow failure consumes search budget.
    Do not let one candidate repeatedly call Codex because it feels close.
    Kill branches that burn budget without new evidence.

Risk: Directly copying SWE-agent commands into Codex.
    Codex already operates in a different tool environment. The right transfer
    is command semantics and state shaping, not necessarily implementing a
    separate `open`/`edit` REPL.

## Local Falsification Tests

1. Context packet ablation:

```text
same task
same model
same evaluator

A: whole relevant file(s)
B: bounded file windows
C: bounded windows plus evaluator digest
D: bounded windows plus reflection from previous failure

Measure:
    pass rate
    changed files
    unrelated edit count
    evaluator runtime
    patch size
```

2. Patch guard ablation:

```text
A: run full evaluator immediately
B: run syntax/import guard first

Measure:
    wasted evaluator calls
    number of candidates rejected before expensive tests
    false rejections
    candidate recovery after guard failure
```

3. History compression ablation:

```text
A: full previous trajectory in next prompt
B: last 5 observations plus compact failure summary
C: only final evaluator report

Measure:
    repeated mistakes
    token cost
    success on retry
```

4. Search summary ablation:

```text
A: raw rg output in prompt
B: capped ranked results
C: oracle file window

Measure:
    localization success
    correct file edited
    irrelevant file edits
```

5. Slow-failure cutoff:

```text
stop candidate when no new file/score/change appears for N iterations
compare against uncapped retries
```

Measure whether early cutoff loses many eventual successes. SWE-agent suggests
successful trajectories tend to finish earlier, but this needs local proof.

## Design Consequences

SWE-agent changes the local architecture from:

```text
evolver:
    pick parent
    ask Codex for patch
    run tests
```

to:

```text
evolver:
    pick parent
    build ACI/context packet
    ask Codex for patch under strict contract
    guard patch
    run visible evaluator
    classify failure
    compact observation
    archive full trajectory
    promote only with hidden/repo-owned gates
```

This is still simple enough to fit in `alpha_evolve/` as microlibs. The mistake
would be to hide all of this inside one agent runner class. The interface pieces
should be testable independently because they are likely to be ablation targets.

## Open Questions For Later Papers

- Does Agentless show that most SWE-agent gains can be kept with an even simpler
  localization plus repair pipeline?
- Does OpenHands add platform lessons that are worth adopting, or is it too
  broad for a local research evolver?
- Can HumanEval/pass@k selection papers give a cheaper first evaluator before
  repo-level patch tasks?
- How should we connect ACI failure labels to AlphaEvolve-style parent selection
  and island diversity?
- Is a Codex batch call enough to implement the SWE-agent interaction loop, or
  do we need a lightweight local agent harness for multi-turn search?
