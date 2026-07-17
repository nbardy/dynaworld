# 007 - Voyager: An Open-Ended Embodied Agent with Large Language Models

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2305.16291
    https://arxiv.org/pdf/2305.16291
    https://voyager.minedojo.org/
    https://github.com/MineDojo/Voyager

Bibliographic metadata:
    Authors: Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar,
    Chaowei Xiao, Yuke Zhu, Linxi Fan, Anima Anandkumar.
    First arXiv submission: 2023-05-25.
    Latest arXiv version inspected: v2, 2023-10-19.

Why this paper matters for alpha_evolve:
    Voyager is the best paper in the queue so far for reusable executable memory.
    LLaMEA and Eureka keep candidate history; Voyager graduates verified code
    into a skill library that future prompts retrieve and compose. For
    DynaWorld, that means `alpha_evolve` should distinguish unverified
    candidates from verified microlib skills. Only artifacts that pass hard
    gates should become reusable prompt material.

One-sentence mechanism:
    An automatic curriculum proposes Minecraft tasks, an action agent writes and
    repairs JavaScript programs using environment feedback and errors, a critic
    verifies completion, and successful programs are embedded into a skill
    library for later retrieval and composition.

## Reading Questions

- What is the executable feedback signal?
  Program execution in Minecraft through Mineflayer, environment chat feedback,
  interpreter errors, and a self-verification critic that decides whether the
  current task succeeded.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Executable action programs. The curriculum also searches over tasks, and the
  skill library searches/retrieves reusable code memories.

- What is the population/database/selection mechanism?
  Voyager is not evolutionary in the population sense. It is a continual
  learning loop: task proposal, up to four repair attempts, verification, skill
  commit on success, failed-task logging on failure, then another curriculum
  task. The skill library is the long-term database.

- What evidence proves the loop improves over one-shot generation?
  Ablations show the skill library prevents late-stage plateau, self-verification
  is the strongest feedback component, and the automatic curriculum is crucial
  for progress. Voyager also reuses the learned library in a new world to solve
  unseen tasks where baselines fail.

- What does the method assume that DynaWorld does not have?
  It assumes open-ended tasks where an LLM critic can often infer success from
  inventory/state. DynaWorld cannot let an LLM critic promote patches or models
  without hard metrics, because visual quality, leakage, and scientific claims
  are easier to misjudge.

## Mechanism

Voyager has three main modules:

```text
automatic curriculum
skill library
iterative prompting mechanism
```

The environment is Minecraft through MineDojo/Mineflayer. The agent does not
output low-level motor commands. It outputs JavaScript functions that call a
small set of control primitives and Mineflayer APIs. Code is the action space.

The paper's pseudocode is:

```text
reset environment
loop:
    summarize exploration progress from completed and failed tasks
    propose next task from current state and progress
    clear code, feedback, errors, critique
    for up to 4 attempts:
        retrieve relevant skills for task and feedback
        generate code from task, previous code, feedback, errors, critique, skills
        execute code in environment
        critic checks task success from state
        break if success
    if success:
        add code to skill library
        mark task completed
    else:
        mark task failed
```

This is not a broad free-form agent. It is a tightly staged loop with different
roles for task proposal, action-code synthesis, verification, and memory.

### Automatic Curriculum

The curriculum agent proposes tasks from:

- current inventory;
- equipment;
- nearby blocks/entities;
- biome;
- time;
- health/hunger;
- position;
- completed tasks;
- failed tasks;
- chest state;
- short task-context Q&A.

The prompt asks for tasks that are challenging but not too hard. It is a
bottom-up curriculum: mine wood before crafting tools, solve local survival
needs before far-away goals, exploit the current biome and inventory.

DynaWorld mapping:

```text
The local equivalent is not a free-form research planner.
It is a bounded problem scheduler over known microlibs and evaluator budgets.
```

A scheduler can choose between:

- retry a failed microlib with a different operator;
- promote a verified helper into the skill library;
- add a heldout test for a suspicious winner;
- move from function-level evolution to bounded patch evolution;
- skip an over-hard problem until prerequisite helpers exist.

But the initial version should probably be static/manual. Voyager's own
ablation says curriculum matters, but its limitations show GPT can propose
impossible tasks.

### Skill Library

Voyager stores successful skills as executable code plus an LLM-generated
description. The description is embedded into a vector database. Future tasks
retrieve top-k relevant skills and include them in the code-generation prompt.

The official repo's checkpoint shape is informative:

```text
skill/
  code/
    catchThreeFishWithCheck.js
    collectBamboo.js
    ...
  description/
    catchThreeFishWithCheck.txt
    collectBamboo.txt
    ...
  skills.json
  vectordb/
```

The `SkillManager` code:

- keeps control primitives separate from learned skills;
- stores learned skill code and description;
- adds descriptions to a Chroma vector database;
- retrieves top-k skills by embedding similarity;
- includes retrieved skill code in future action prompts;
- supports resume from a skill library directory.

This is the strongest concrete mechanism for local `alpha_evolve`: a candidate
archive is not the same as a skill library.

For DynaWorld:

```text
candidate archive = every attempt, including failures
skill library = only verified reusable artifacts
```

The skill library should not store full arbitrary patches by default. It should
store small reusable helpers, prompt compressors, evaluators, score-signature
builders, context pruners, and bounded patch patterns that have stable
contracts.

Proposed local skill record:

```json
{
  "skill_id": "uuid",
  "name": "summarize_component_traces",
  "description": "Builds compact reflection from component trend metrics.",
  "artifact_kind": "python_helper",
  "code_path": "skills/code/summarize_component_traces.py",
  "contract_path": "skills/contracts/summarize_component_traces.md",
  "verified_on": ["unit_component_traces", "leak_status_fixture"],
  "score_signature": {"passed_stage": "unit", "finite": true},
  "api_dependencies": ["score_signature.v1"],
  "invalidates_on": ["score_schema_change"],
  "parent_candidate_ids": ["..."],
  "embedding_text": "..."
}
```

The hard rule: do not retrieve and reuse unverified code as a skill.

### Iterative Prompting

Voyager repairs code over multiple attempts. The action prompt includes:

- code-generation guidelines;
- control primitive APIs;
- retrieved skills;
- previous generated code;
- environment feedback/chat logs;
- execution errors;
- critic critique;
- current state;
- task;
- task context;
- reasoning before code.

This is the practical shape of a repo repair loop:

```text
previous patch
stderr/stdout/log summary
hard gate result
critic/human critique if any
retrieved verified helpers
current task contract
new patch
```

Voyager gives the action agent up to four attempts before moving on. That is a
useful local default. Infinite repair on a doomed candidate wastes budget and
pollutes memory. After N failed attempts, record the failure and let the
curriculum/scheduler decide whether to revisit later.

### Self-Verification

Voyager uses a critic agent to decide whether a task succeeded. The critic sees
state such as inventory, nearby blocks, equipment, health, hunger, chest state,
task, and task context. It outputs JSON with reasoning, success, and critique.

The ablation says self-verification is the most important feedback type. Removing
it drops discovered item count by 73 percent. That is a strong result, but it
does not transfer literally to DynaWorld.

For DynaWorld:

```text
LLM critic = diagnosis and qualitative critique
hard evaluator = promotion and acceptance
human critique = optional qualitative reflection
```

An LLM can say "this video still looks smeared" or "the change probably edits
the wrong abstraction." It should not set `passed=true` for a training or
renderer claim. That belongs to the evaluator cascade and baseline docs.

### Human Feedback

Voyager demonstrates human feedback in two equivalent roles:

- human as critic, replacing self-verification with visual/spatial critique;
- human as curriculum, breaking a complex task into subgoals.

This maps well to DynaWorld because some important judgments are visual or
strategic. A human can say:

```text
This candidate improved numeric score but lost object identity.
Try a smaller helper rather than changing the trainer loop.
Do not chase this route; schedule heldout sampler work first.
```

That should be stored as reflection. It should not silently become a changed
metric.

## Evaluation

Voyager evaluates in Minecraft/MineDojo using GPT-4 for action code,
curriculum, and critic, GPT-3.5 for lower-cost Q&A/description support, and
OpenAI embeddings for skill retrieval. Temperatures are mostly 0; curriculum
uses a small nonzero temperature for diversity.

Baselines:

- ReAct adapted to MineDojo;
- Reflexion adapted with execution errors and self-verification;
- AutoGPT-style subgoal decomposition;
- Voyager without skill library;
- design ablations for curriculum, feedback types, and model quality.

Main results:

- 3.3x more unique items than baselines within 160 prompting iterations;
- 2.3x longer map traversal;
- key tech tree milestones up to 15.3x faster;
- only Voyager reaches the diamond tool level in the reported tech tree table;
- skill library transfers to a new world for unseen tasks;
- AutoGPT also improves when given Voyager's learned skill library.

Ablation results relevant to `alpha_evolve`:

- replacing automatic curriculum with random curriculum drops discovered item
  count by 93 percent;
- removing the skill library causes later-stage plateau;
- removing self-verification drops discovered item count by 73 percent;
- GPT-4 code generation discovers 5.7x more unique items than GPT-3.5 in the
  reported ablation;
- skill retrieval top-5 accuracy is reported as 96.5 percent over 309 samples.

The main evidence for us is the transfer result: a learned executable library is
useful beyond the exact world/task where it was created. That is what a local
microlib skill library should aim for.

## Failure Modes

### Skill Pollution

If a bad skill is committed, future prompts may retrieve and compose it. This is
worse than a bad candidate because it becomes reusable prior context.

DynaWorld rule:

```text
Skill promotion requires stricter gates than candidate selection.
```

Candidates can be interesting with partial scores. Skills need stable contracts,
passing tests, known dependencies, and invalidation rules.

### LLM Critic False Positives

Voyager notes that self-verification can fail. In Minecraft, a false positive
may commit a bad skill. In DynaWorld, a false positive could promote a broken
training path or misleading metric.

DynaWorld rule:

```text
LLM critic output is reflection, not acceptance.
```

### Curriculum Hallucination

Voyager's curriculum can propose nonexistent tasks or invalid items. In
DynaWorld, an unconstrained curriculum could propose refactors outside the
current research lane or tasks with no cheap evaluator.

DynaWorld rule:

```text
Curriculum choices must be from a finite task registry until the evaluator
system is mature.
```

### API Hallucination

Voyager sees generated code call nonexistent APIs or invalid resources. The
local equivalent is Codex calling helper functions that do not exist, changing
config keys that are not normalized, or assuming data paths that are absent.

DynaWorld rule:

```text
Action prompts should include allowed API surfaces and the evaluator should
fail missing-symbol/API hallucinations early.
```

### Retrieval Mismatch

Embedding retrieval can return a plausible but unsafe skill. The paper reports
high top-5 accuracy, but DynaWorld skills may have compatibility constraints
that semantic similarity does not capture.

DynaWorld rule:

```text
Retrieve by embedding plus structured filters: mode, schema version, allowed
paths, data contract, evaluator family, and dependency version.
```

### Cost And Model Dependence

Voyager depends heavily on GPT-4-quality code generation. It is expensive, and
GPT-3.5 underperforms strongly.

DynaWorld rule:

```text
Use the strongest Codex model for expensive patch evolution. Use smaller models
only for retrieval summaries, candidate tagging, and cheap reflection.
```

## DynaWorld Mapping

### `verified_skill_library`

Add a first-class skill library to the `alpha_evolve` design, separate from
candidate archives.

Proposed layout:

```text
alpha_evolve/
  skills/
    code/
    contracts/
    descriptions/
    skills.jsonl
    vectordb/
    invalidation_rules.jsonc
```

The library should store only artifacts that pass a promotion gate:

```text
candidate passes evaluator
candidate has stable contract
candidate has narrow API dependencies
candidate has no hidden-data access
candidate has useful transfer value
```

Transfer value matters. A one-off patch that only wins a single benchmark should
stay in the candidate archive. A helper that improves many prompt packs or
evaluators can become a skill.

### `skill_retriever`

Voyager retrieves top-5 skills by embedding similarity. DynaWorld should do a
hybrid retrieval:

```text
semantic query:
  task text + failure reflection + score signature

structured filters:
  microlib kind
  allowed paths
  evaluator stage
  schema version
  data mode: same_view, heldout_view, mixed
  artifact kind: helper, patch pattern, prompt compressor, scorer
```

The prompt should include skill descriptions and code only when the skill is
compatible with the current contract. Otherwise retrieval becomes a leak path.

### `repair_attempt_loop`

Voyager's four-attempt loop is directly useful:

```text
for attempt in max_attempts:
    build prompt with previous patch, logs, critique, retrieved skills
    run codex exec
    apply in candidate worktree
    run evaluator stage
    if hard pass: break
record failed attempt sequence if not pass
```

This loop belongs inside candidate generation, before archive promotion. It is
not the same as evolutionary generations. One generation can contain multiple
repair attempts to make one candidate runnable.

### `task_scheduler`

Voyager's automatic curriculum should start as a bounded scheduler:

```text
task registry:
  score_signature_helper
  prompt_pack_compressor
  context_pruner
  component_trace_logger
  config_cleanup_microrefactor
  sampler_leakage_audit_helper
  renderer_microbench_policy

state:
  completed tasks
  failed tasks
  verified skills
  current evaluator cost budget
  recent failure histogram
```

The scheduler can propose the next problem, but only from the registry. It can
become more open-ended later if it proves it does not hallucinate impossible
work.

### `critic_agent`

A local critic can be useful if its output is limited:

```text
allowed:
  summarize failure
  point to likely abstraction mismatch
  suggest next repair focus
  flag suspicious metric wins
  translate human qualitative feedback into reflection text

forbidden:
  mark hard pass
  change hidden evaluator
  update baseline table
  promote a skill without tests
```

This preserves Voyager's most useful feedback path without importing its
success-checking risk.

## Falsification Tests

### Test 1: Skill Library Transfer

Build a synthetic microlib suite where several tasks require reusable helper
patterns. Compare:

```text
candidate evolution with no skill retrieval
candidate evolution with verified skill retrieval
candidate evolution with unverified archive retrieval
```

Claim falsified if verified skills do not improve solve rate or reduce attempts
relative to no retrieval. Also fail the design if unverified retrieval improves
best score but increases regression rate.

### Test 2: Skill Pollution Guard

Inject a skill with a plausible description and a subtle contract violation.

Claim falsified if the retriever includes it in a prompt where structured
filters should reject it, or if the promotion gate lets it into the skill
library.

### Test 3: Critic False Positive Containment

Make the LLM critic claim success for a candidate that fails the hard evaluator.

Claim falsified if the runner promotes the candidate or records the skill as
verified.

### Test 4: Curriculum Registry Constraint

Ask the scheduler to propose next tasks from state containing several failures.

Claim falsified if it proposes a task outside the registry or one without an
evaluator.

### Test 5: Retry Budget

Create a candidate that repeatedly fails for the same missing API. Run the
repair loop with max attempts.

Claim falsified if the runner keeps spending attempts after the cap or does not
record a structured failed-task entry.

### Test 6: Retrieval Compatibility

Create skills for same-view, heldout-view, and mixed-view modes. Query from each
mode.

Claim falsified if a heldout-only skill is retrieved for a same-view contract or
if a skill with a stale schema version is included in a prompt.

## Notes For Future Papers

- ReAct should be read as the baseline that Voyager beats by adding curriculum,
  skill library, and verification.
- Reflexion should be compared against Voyager's critic: reflection alone is not
  the same as task-success verification.
- SWE-agent and OpenHands should be checked for how repo agents handle reusable
  tools and memory without polluting future tasks.
- CodeT should be read with Voyager in mind: generated tests can become skills,
  but only after hard validation.

## Bottom Line

Voyager upgrades the `alpha_evolve` plan from "archive all candidates" to a
two-layer memory system:

```text
candidate archive:
  every attempt, every failure, every score

verified skill library:
  only reusable code artifacts that passed promotion gates
```

For DynaWorld, this is the right way to reuse Codex discoveries without letting
bad patches become future priors. The first version should build skill-library
infrastructure around small microlibs, not whole-trainer patches.
