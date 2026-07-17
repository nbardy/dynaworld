# 016 - OpenHands: An Open Platform for AI Software Developers as Generalist Agents

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2407.16741
    https://arxiv.org/pdf/2407.16741
    https://github.com/All-Hands-AI/OpenHands
    https://github.com/OpenHands/OpenHands
    https://docs.openhands.dev/

Implementation artifacts inspected:
    https://github.com/OpenHands/OpenHands/blob/main/README.md

Bibliographic metadata:
    Authors: Xingyao Wang, Boxuan Li, Yufan Song, Frank F. Xu, Xiangru Tang,
    Mingchen Zhuge, Jiayi Pan, Yueqi Song, Bowen Li, Jaskirat Singh, Hoang H.
    Tran, Fuqiang Li, Ren Ma, Mingzhang Zheng, Bill Qian, Yanjun Shao, Niklas
    Muennighoff, Yizhe Zhang, Binyuan Hui, Junyang Lin, Robert Brennan, Hao
    Peng, Heng Ji, Graham Neubig.
    First arXiv submission: 2024-07-24.
    Version inspected: arXiv v3, 2025-04-18.
    Venue/context: ICLR 2025.

Current implementation note:
    The current README has evolved beyond the paper-era OpenDevin framing. It
    now describes OpenHands as AI-driven development with SDK, CLI, local GUI,
    cloud, and enterprise surfaces. It also points to a newer Software Agent SDK
    technical report. This note focuses on the 2025 OpenHands paper, but flags
    that implementation architecture should be checked against current docs
    before copying details.

Why this paper matters for alpha_evolve:
    OpenHands is not the first local implementation target. Agentless argues for
    a staged baseline before platform complexity. But OpenHands is useful as an
    infrastructure checklist: if `alpha_evolve/` becomes more than a few batch
    Codex calls, it needs an event stream, action/observation schema, sandbox
    runtime, skill registry, benchmark registry, cost tracking, and quality
    control tests.

    The local lesson is:

```text
do not start with a giant agent platform
borrow the small platform primitives that make experiments reproducible
```

One-sentence mechanism:
    OpenHands provides a general agent platform where agents read an event
    stream and emit actions, a Docker-backed runtime executes bash/Python/browser
    actions and returns observations, skills extend the agent-computer
    interface, delegation composes agents, and a benchmark framework evaluates
    agents across software, web, and assistance tasks.

## Reading Questions

- What is the executable feedback signal?
  OpenHands itself is a platform, so feedback depends on the integrated
  benchmark or task. At the platform level, feedback is represented as
  observations returned after actions. At the evaluation level, feedback is task
  metrics from SWE-bench, HumanEvalFix, WebArena, MiniWoB++, GPQA, GAIA, and
  other benchmarks.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  The platform supports trajectory search or agent execution, but does not by
  itself define an evolutionary search. The search unit is the action/observation
  event stream produced by an agent. OpenHands can host agents, microagents, and
  delegated subagents whose policies can be compared.

- What is the population/database/selection mechanism?
  No population database in the AlphaEvolve sense. The reusable artifact is the
  event stream plus benchmark evaluation framework. For local evolution, those
  become the candidate trajectory database and evaluation harness.

- What evidence matters?
  OpenHands reports that CodeActAgent v1.8 achieves competitive but not top
  SWE-bench Lite performance: 7.0 percent with GPT-4o-mini, 22.0 percent with
  GPT-4o, and 26.0 percent with Claude 3.5 Sonnet in the paper table. It also
  evaluates across 15 benchmarks, emphasizing generality over leaderboard
  specialization.

- What does this assume that DynaWorld does not yet need?
  A full GUI, hosted cloud surface, arbitrary web browsing, multi-user
  collaboration, and broad benchmark integration. DynaWorld needs a much smaller
  internal runner. Copying OpenHands wholesale would add too much product and
  infrastructure surface before the research loop is clear.

## Platform Architecture

OpenHands has three core pieces:

```text
agent abstraction:
    state -> action

event stream:
    chronological action/observation history
    user interactions
    agent messages
    cost and metadata

runtime:
    sandbox executes actions
    action result becomes observation
```

The paper's model:

```text
agent reads event history
agent produces action
runtime executes action
runtime emits observation
event stream stores observation
agent reads updated state
```

For `alpha_evolve`, the event stream is the relevant primitive. Today a batch
Codex candidate can look like a black box:

```text
prompt in
diff out
test log out
```

OpenHands suggests a richer but still structured record:

```text
candidate_id
prompt message
action type
action payload
observation type
stdout/stderr/artifacts
cost
timestamp
parent/delegation metadata
final patch
final score
```

This is useful even if we do not run an interactive OpenHands-style agent.

## Action Space

The paper's core actions include:

```text
IPythonRunCellAction:
    execute Python/IPython code in the sandbox

CmdRunAction:
    execute bash commands in the sandbox

BrowserInteractiveAction:
    interact with a browser through BrowserGym/Playwright-style primitives

MessageAction:
    communicate with user or other agents

AgentFinishAction:
    end task

AgentDelegateAction:
    delegate a subtask to another agent
```

This action space is intentionally general and programmer-like. Agents can write
code, execute shell commands, use browser actions, and communicate.

Local transfer:

```text
alpha_evolve does not need broad browser action support at first
alpha_evolve does need typed action/observation records
alpha_evolve may need CmdRun and PatchApply actions
alpha_evolve may later need DelegateAction for expensive sidecar evaluation
```

A smaller local action schema:

```text
BuildContextAction
CodexExecAction
PatchParseAction
PatchApplyAction
RunGuardAction
RunEvaluatorAction
ClassifyFailureAction
PromoteCandidateAction
ArchiveAction
```

The runner can execute these actions deterministically rather than letting the
LLM choose them.

## Runtime

OpenHands uses a Docker sandbox per task session. The sandbox contains:

```text
bash shell
Jupyter/IPython server
Chromium browser based on Playwright
workspace mount
OpenHands action execution API
```

The runtime runs an API server inside the Docker container. The backend sends
actions over REST, the sandbox executes them, and observations return to the
event stream.

Appendix runtime workflow:

```text
1. user provides base Docker image
2. OpenHands builds a runtime image with runtime client/API code
3. container launches
4. backend sends actions to runtime client
5. runtime executes actions in sandbox
6. runtime returns observations to event stream
```

It also uses a dual tagging system:

```text
hash tag:
    reproducible image contents

generic tag:
    stable reference to latest version for a base image / OpenHands version
```

Local transfer:
    DynaWorld already has expensive environment concerns. The first
    `alpha_evolve` runner can use git worktrees or temp dirs instead of Docker,
    but the runtime boundary should be explicit:

```text
runtime adapter:
    local worktree
    temp copy
    Docker container
    Modal/cloud later
```

The candidate database should record runtime identity. A score produced in a
dirty local worktree is not equivalent to a score produced in a pinned container.

## AgentSkills

OpenHands includes an AgentSkills library imported into the IPython environment.
It does not try to wrap every Python package. The inclusion rule is:

```text
add a skill when the task is not readily achievable for an LLM to write directly
or when it calls an external model/service
```

Examples include file editing utilities adapted from SWE-agent/Aider, scrolling
helpers, image parsing, and PDF parsing.

This is a good rule for local microlibs. Do not wrap `rg`, `pytest`, or simple
filesystem reads in elaborate abstractions. Add a microlib when it:

```text
prevents common model mistakes
normalizes an evaluator contract
captures artifact provenance
reduces expensive repeated work
exposes a hard-to-write operation safely
```

For `alpha_evolve`, skills should be implementation modules, not prompts only:

```text
patch_parser
score_parser
context_packet_builder
candidate_archive_writer
runtime_fingerprint
generated_test_sandbox
```

## Delegation

OpenHands supports `AgentDelegateAction`, where one agent can delegate a subtask
to another specialized agent. The paper gives the example of a generalist
CodeActAgent delegating browsing tasks to a BrowsingAgent.

Local transfer:
    Do not start with multi-agent delegation. But the event schema should not
    preclude it. Expensive future subtasks could be delegated:

```text
localizer:
    build context packet

patcher:
    generate candidate patch

verifier:
    run expensive smoke/evaluator

critic:
    classify failure

ranker:
    pick final candidate
```

The first implementation can run these roles as deterministic functions or
separate Codex calls. Only promote them to agent workers when there is measured
benefit.

## AgentHub, Microagents, And Prompts

OpenHands includes multiple agent implementations and describes microagents:
small specialized agents that reuse most of a generalist agent but add task
specific prompts/behavior.

For DynaWorld:

```text
microagent is close to microlib plus prompt profile
```

Potential local profiles:

```text
config_cleanup_agent:
    knows AGENTS.md P1-P5 cleanup rules

benchmark_harness_agent:
    knows how to preserve baseline docs and result artifacts

renderer_guard_agent:
    knows which smoke/verifier gates matter for renderer changes

paper_synthesis_agent:
    knows note format and source discipline
```

But these should be thin profiles over the same staged runner, not independent
frameworks.

## Evaluation Framework

OpenHands integrates 15 benchmarks across:

```text
software engineering:
    SWE-bench Lite
    HumanEvalFix
    ML-Bench
    BioCoder
    Gorilla APIBench
    BIRD

web browsing:
    WebArena
    MiniWoB++

miscellaneous assistance:
    GPQA
    GAIA
    AgentBench
    MINT
    ToolQA
    and related tasks
```

The point is not the exact benchmark list. The point is a registry where tasks
share:

```text
task loader
runtime setup
agent invocation
metric extraction
artifact capture
cost tracking
result table
```

Local transfer:

```text
alpha_evolve/evaluators/
    registry.py
    anti_pattern_cleanup.py
    config_smoke.py
    benchmark_harness.py
    generated_test.py
    hidden_gate.py
```

Each evaluator should declare:

```text
name
description
command
timeout
runtime requirements
visible_to_candidate
promotion_eligible
score parser
artifact paths
```

This keeps generated tests, cheap visible smokes, and hidden promotion gates
separate.

## Results

OpenHands is framed as a generalist platform, not a top single benchmark
specialist.

Selected SWE-bench Lite results from the paper:

```text
SWE-agent with GPT-4-1106-preview:
    18.0 percent

AutoCodeRover with GPT-4-0125-preview:
    19.0 percent

Aider with GPT-4o and Claude 3 Opus:
    26.3 percent

Agentless with GPT-4o:
    27.3 percent in the OpenHands table

OpenHands CodeActAgent v1.8 with GPT-4o-mini:
    7.0 percent

OpenHands CodeActAgent v1.8 with GPT-4o:
    22.0 percent

OpenHands CodeActAgent v1.8 with Claude 3.5 Sonnet:
    26.0 percent
```

HumanEvalFix:

```text
SWE-agent:
    87.7 percent, 1-shot demonstration

OpenHands CodeActAgent v1.5:
    79.3 percent, generalist, 0-shot
```

The paper argues that the same CodeAct agent is competitive across software,
web, and general-assistance categories without task-specific prompt changes.
For `alpha_evolve`, that is a tradeoff:

```text
generality:
    useful if the target set includes code, docs, browser, and data tasks

specialization:
    likely better for first DynaWorld microlib evolution tasks
```

Agentless remains a stronger first baseline for SWE-style code repair. OpenHands
is stronger as a platform design reference.

## Quality Control

The paper describes an end-to-end agent test framework because full benchmark
runs are slow and expensive. It notes that a SWE-bench Lite evaluation with
GPT-4o costs around 600 dollars.

Local transfer:
    We need quality-control smokes for the `alpha_evolve` runner itself:

```text
can build context packet
can call a fake Codex backend
can parse a patch
can apply patch in temp runtime
can run dummy evaluator
can archive trajectory
can reject malformed output
can resume after interrupted candidate
```

These are not candidate-fitness tests. They are runner health tests. Without
them, evaluator regressions will be confused with candidate regressions.

## What Transfers To `codex exec`

OpenHands can be too much for a local AlphaEvolve proof of concept, but its
event-stream architecture maps cleanly to batch Codex calls:

```text
CodexExecAction:
    prompt
    model/profile args
    cwd/runtime id
    allowed file scope

CodexExecObservation:
    stdout/stderr
    changed files
    duration
    exit status
    token/cost estimate when available
```

The outer runner can then add deterministic actions:

```text
BuildContextAction
PatchGuardAction
RunEvaluatorAction
ClassifyFailureAction
ArchiveCandidateAction
```

The point is not to implement a chat UI. The point is to make every step
replayable and inspectable.

## Microlibs Suggested By This Paper

```text
event_stream/
    Append-only JSONL actions and observations per candidate.

action_schema/
    Typed records for context, Codex, patch, evaluator, ranking, archive.

runtime_adapter/
    Local temp dir, git worktree, Docker, or remote runtime behind one interface.

sandbox_workspace/
    Prepare isolated candidate workspace and collect diff/artifacts.

skill_registry/
    Small reusable helpers exposed to prompts/runners only when needed.

evaluator_registry/
    Declares visible/hidden gates, commands, parsers, timeouts, artifacts.

benchmark_runner/
    Runs a task suite across runner variants and produces comparable tables.

cost_tracker/
    Tracks Codex calls, evaluator runtime, and expensive hidden gates.

delegate_schema/
    Represents optional future worker/subtask delegation without requiring it.

runner_qc/
    Cheap tests for the evolver infrastructure itself.
```

## Target Problems In This Repo

OpenHands-style infrastructure matters once we compare runner variants across
more than one task. Candidate first suites:

```text
anti-pattern cleanup suite:
    needs evaluator registry and task loader

config normalization suite:
    needs isolated worktree and hidden smoke gates

benchmark-harness repair suite:
    needs artifact capture and command replay

paper-note synthesis suite:
    needs source capture, note schema, and weaker evaluator labels

browser/web tasks:
    not first priority for DynaWorld alpha_evolve
```

The local rule:

```text
if a task can be done with Agentless-style staged patching, do that first
if a task needs stateful shell/browser exploration, borrow OpenHands runtime ideas
```

## Red-Team Notes

Risk: Platform gravity.
    A general platform can consume engineering time before the research question
    is tested. Keep the first runner small.

Risk: Generalist agent underperforms specialized baseline.
    The paper itself shows competitive but not dominant SWE-bench Lite results.
    Do not use OpenHands-like generality as evidence that an agentic loop beats
    Agentless-style staged repair.

Risk: Sandbox mismatch.
    Docker improves reproducibility but can differ from local GPU/Metal
    training environments. Record runtime identity and do not mix scores.

Risk: Event logs without semantics.
    A full event stream is useful only if actions and observations are typed
    enough to compute failure labels and replay evaluations.

Risk: Skill sprawl.
    AgentSkills has an inclusion philosophy. Local microlibs need one too:
    add helpers that prevent model mistakes or normalize contracts, not wrappers
    for everything.

Risk: Delegation before measurement.
    Multi-agent delegation should be a later optimization, not the first design.

## Local Falsification Tests

1. Event-stream necessity:

```text
A: archive only prompt/diff/final score
B: archive typed actions/observations

Measure:
    can we classify failures?
    can we reproduce candidate?
    can we compare runner variants?
```

2. Runtime isolation:

```text
A: run candidates in current worktree
B: run candidates in temp copy/worktree

Measure:
    dirty-state leakage
    reproducibility
    cleanup failures
```

3. Evaluator registry:

```text
A: ad hoc commands embedded in task prompt
B: evaluator registry with visible/hidden flags

Measure:
    accidental hidden leakage
    result comparability
    task authoring cost
```

4. Runner QC:

```text
intentionally malformed patch
intentionally failing evaluator
interrupted candidate
missing artifact
```

Verify the runner fails loudly and archives the failure.

5. Platform cutoff:

```text
compare implementation time and pass rate:
    staged Agentless runner
    OpenHands-style interactive runner
```

Use the interactive runner only if it solves measured failure classes that the
staged runner misses.

## Design Consequences

OpenHands should influence the local folder shape only after the Agentless
baseline exists:

```text
alpha_evolve/
    tasks/
    context/
    localize/
    patch/
    validate/
    rank/
    archive/
    runtime/
    events/
    evaluators/
    qc/
```

The key new additions from OpenHands are `events/`, `runtime/`, `evaluators/`,
and `qc/`.

Do not start by implementing:

```text
GUI
browser integration
cloud execution
multi-user state
enterprise auth
full agent marketplace
```

Those are product/platform features, not first-order evidence for whether
Codex-driven evolution improves DynaWorld code.

## Open Questions For Later Papers

- Can HumanEval/pass@k theory tell us how many patch samples are enough before
  ranker improvements matter more?
- Can AlphaCode-style massive sampling be adapted to repo patches without
  exploding evaluator cost?
- Does CodeT provide a safer generated-test selection method than Agentless's
  reproduction tests?
- Should the first runtime be temp directories, git worktrees, Docker, or a
  pluggable adapter supporting all three?
- What runner QC suite is sufficient before spending real model/evaluator
  budget?
