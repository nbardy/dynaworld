# Cross-Paper Synthesis

This file should be updated after every 3-5 paper notes.

## Current Working Thesis

For DynaWorld, "agentic evolution" is only useful when the unit of evolution is
small enough to be judged by a hard evaluator. The likely winning architecture
is not a broad multi-agent swarm. It is:

```text
microlib contract
candidate worktree
Codex patch
evaluator cascade
program database
prompt sampler with prior candidates and failure notes
```

The deepest risk is metric hacking. Almost every paper in the queue should be
read through that lens: what prevents the model from satisfying the evaluator
while missing the scientific or engineering intent?

## After Papers 001-002

AlphaEvolve says the long-term local shape can evolve larger code regions.
FunSearch says the first proof should be much narrower: fixed skeleton, one
evolved callable or helper, evaluator, candidate archive, score signatures, and
best-shot prompts from two prior candidates. The practical sequence should be:

```text
function-level microlib
helper-file microlib
strict allowed-path patch microlib
larger trainer/kernel patch only after the archive loop beats one-shot Codex
```

This changes the local design slightly. `programs.jsonl` should store a score
signature, not only a scalar score dict, and the prompt sampler should show
contrasting measured candidates from the same island rather than only the global
best. For DynaWorld, signatures should encode behavior that we do not want a
single scalar to erase: batch-kind coverage, leakage status, finite status,
gradient presence, overflow status, timing bucket, and changed-LOC bucket.

## After Paper 003

CodeEvolve fills the implementation gap between FunSearch and AlphaEvolve. The
local runner should now be designed as two modes:

```text
mode 1: skeleton/function evolution
mode 2: bounded patch evolution
```

Mode 1 is the first proof target because it is easier to evaluate. Mode 2 is
what DynaWorld ultimately needs for trainer, renderer, and config work. The
database schema should therefore not assume only function text or only patch
diffs; it should store both as candidate artifacts, plus parent IDs, prompt IDs,
ancestor IDs, inspiration IDs, operator name, island, migration origin, archive
cell, highest evaluator stage passed, and failure-log path.

Prompt evolution is useful but dangerous in this repo. A prompt should have an
immutable contract section and a mutable strategy section. Meta-prompting may
rewrite the strategy; it must not rewrite allowed paths, forbidden edits,
evaluator commands, data-contract language, or final response schema.

The smallest CodeEvolve-shaped runner worth building is probably:

```text
3 islands
cycle migration
ancestor_depth = 3
inspirations = 2
exploration_rate = 0.2
behavioral archive descriptors chosen per microlib
```

This is not a claim that these numbers are optimal. It is the smallest shape
that preserves the mechanisms CodeEvolve's ablations say matter: history,
inspirations, island diversity, migration topology, and behavior-preserving
archives.

## After Paper 004

Evolution of Heuristics adds the missing operator taxonomy. The candidate
database should not only store patches and metrics; it should store a short
candidate `thought` and the operator that produced it:

```text
explore_different
extract_common_backbone
improve_parent
retune_constants
simplify_patch
```

This is a practical prompt scheduler for early `codex exec` evolution. E1/E2
create diversity, M1/M2 exploit working candidates, and M3 fights the code bloat
that evolutionary systems naturally accumulate. The thought is not evidence;
the evaluator remains evidence. But thoughts are useful compressed prompt
material when they match the diff and metrics.

EoH also changes seeding policy: local evolution should begin from expert seeds
when available. For DynaWorld, an expert seed is the current hand-written
implementation, benchmark-positive prototype, or best near miss with a clear
failure note. Blank-slate creativity is less valuable than measured improvement
over a known baseline.

## After Paper 005

LLaMEA changes the implementation order. Before building a full islanded
CodeEvolve clone, build the smallest falsifiable serial runner:

```text
llamea_serial
one microlib
one selected parent
one compact archive summary
codex exec mutation
hard evaluator cascade
structured error feedback
best_so_far/latest_parent selection
```

The key mechanism is not population size. It is the prompt state: immutable task
contract, names/scores/history summary, selected parent code or patch, measured
score/std/signature, and concrete error feedback. That is enough to test whether
evolution beats independent `codex exec` sampling on the same call budget. The
goal shorthand said `codex -p`, but the current local CLI uses `-p` for
`--profile`; the runner should hide the exact CLI shape behind a Codex adapter.

LLaMEA also adds an evaluator design lesson: score curves matter. Its AOCC
metric rewards anytime performance rather than only the final value. DynaWorld
microlibs should use AOCC-like staged scores where useful: syntax/import pass,
unit pass, smoke pass depth, quality curve over fixed train/render steps, time
penalty, and regression penalty. The prompt should see the scalar plus compact
score signature; the database should keep the full vector.

Error feedback should be stored as candidate data, not treated as runner noise.
The local candidate schema needs at least `stage_failed`, command, return code,
timeout, exception summary, stderr tail, log path, and selection score. Failed
candidates are often the best next prompt examples.

The maintained LLaMEA library has useful later features: diff mode, HPO,
niching, population evaluation, timeouts, parallel evaluation, and adaptive
prompt/mutation controls. Treat these as a roadmap after the serial loop proves
signal, not as first-version requirements.

## After Paper 006

Eureka adds the missing metric-hacking boundary:

```text
generated code may shape training
generated code may summarize diagnostics
generated code may propose visible tests
generated code must not edit hidden acceptance
```

The transferable mechanism is the separation between generated reward `R` and
external fitness `F`. Eureka evolves reward code, trains a policy under that
reward, then selects by an external task metric. For DynaWorld, this means any
Codex-evolved loss, score compressor, curriculum, diagnostic, or generated test
must remain subordinate to an immutable evaluator and heldout data contract.

Eureka also upgrades the feedback design. Generated reward code must expose
named components, and the runner should log component traces over checkpoints.
The next prompt should get reflection like: component flat, component dominates,
task score near zero, hidden leak failed, execution crashed. That is more useful
than a scalar alone and safer than dumping full logs.

This suggests three early microlibs before full reward/loss evolution:

```text
context_pruner
component_trace_logger
reward_reflection_builder
```

All three are cheap, testable, and directly reduce the risk that a future
evolution loop leaks hidden state or optimizes a broken proxy.

## After Paper 007

Voyager adds a memory split that should become a hard local invariant:

```text
candidate archive = every attempt, including failures
verified skill library = only reusable artifacts that passed promotion gates
```

This changes how prior candidates should enter prompts. The archive can provide
score signatures, failure summaries, and contrasting parents. The skill library
can provide executable helper code. Mixing those two is dangerous: an
unverified near miss should not become reusable prior context just because it
looked promising.

Voyager also suggests a repair-loop layer inside one evolutionary generation:

```text
max_attempts = 4
prompt = previous patch + logs + hard gate result + critic/human critique + retrieved skills
run codex exec
evaluate in candidate worktree
stop on hard pass or record structured failed task
```

The critic is useful, but not authoritative. In this repo, LLM or human critique
can diagnose failures and steer the next prompt; hard evaluator stages decide
promotion. That keeps Voyager's feedback benefits without importing its biggest
risk: false-positive self-verification.

Early skill-library microlibs:

```text
verified_skill_library
skill_retriever
repair_attempt_loop
bounded_task_scheduler
critic_reflection_adapter
```

## After Paper 008

ReAct defines the simplest baseline loop underneath the heavier systems:

```text
thought
action
observation
repeat
```

For local evolution, this means a candidate generation step should have an
auditable trace before it has an archive or island. The trace should state what
the agent believed, what it did, and what evidence came back. Thoughts and
reflections are useful control surfaces, but they are not evidence; evaluator
artifacts remain evidence.

ReAct also creates a baseline that evolution must beat:

```text
one-shot codex exec
react_repair_loop with max_attempts
llamea_serial with same Codex-call budget
```

If `llamea_serial` does not beat a bounded ReAct repair loop, the problem does
not yet justify program databases or islands. This matters because many repo
fixes are repair problems, not discovery problems.

Local microlibs added by ReAct:

```text
react_repair_loop
candidate_trace_schema
loop_detector
observation_provenance_checker
human_reflection_store
```

## After Paper 009

Reflexion adds the memory layer that should sit inside the ReAct repair loop and
inside each evolutionary branch:

```text
visible evaluator feedback -> compact reflection -> next Codex attempt
```

The hard boundary is:

```text
reflection = hypothesis about failure
evaluator = judge
archive = evidence
verified skill = promoted reusable artifact
```

Do not let reflection become truth. It can make retries less repetitive, but it
must be keyed by problem id, evaluator fingerprint, parent family, and visible
score signature. It also needs invalidation when the evaluator or data contract
changes. The most dangerous failure is memory poisoning: one confident but wrong
reflection becomes inherited by a whole candidate family.

Reflexion's programming result also sharpens the generated-test boundary.
Generated tests are useful for repair and visible feedback, but false positives
are worse than false negatives. A false negative costs another attempt; a false
positive can promote a bad candidate. For DynaWorld, generated tests and cheap
smokes can shape `codex exec` attempts, but promotion belongs to repo-owned
gates and heldout contracts.

Local microlibs added by Reflexion:

```text
verbal_reflection_memory
reflection_builder
reflection_invalidator
false_positive_guard
reflection_budget_controller
```

This changes the staged baseline:

```text
one-shot codex exec
react_repair_loop + structured reflection memory
llamea_serial with reflection summaries
program database / islands with reflection utility tracking
```

Reflexion is weak at novelty. The WebShop negative result says that repeated
reflection does not automatically escape local minima when the target requires
diverse exploration. Use reflection inside a branch; use islands, novelty
descriptors, or search when repeated reflection failures show no score delta.

## After Paper 010

Self-Refine is the small local loop:

```text
generate -> feedback -> refine -> repeat
```

It should be a helper, not the main judge. The useful transfer is prompt
discipline: feedback must be specific, actionable, grounded in visible evidence,
and cheap enough that it does not consume all candidate budget. The dangerous
transfer is same-model self-critique without execution; that is exactly where
math gains were weak and where code feedback can point at the wrong location or
suggest the wrong fix.

For DynaWorld, Self-Refine fits soft and prompt-like artifacts first:

```text
prompt templates
failure summaries
generated-test wording
candidate metadata schema
post-pass readability
```

For actual code mutation, use the safer hybrid:

```text
deterministic evaluator facts + model feedback -> Codex refine attempt
```

not model feedback alone. The paper's failure analysis says most unsuccessful
cases come from bad feedback, not from failing to implement good feedback. That
makes `feedback_quality` a first-class metric in the evolver.

Local microlibs added by Self-Refine:

```text
self_refine_loop
feedback_actionability_scorer
refinement_history_selector
soft_quality_refiner
oracle_feedback_adapter
```

One important local deviation: the paper returns the latest refinement, but
evolution should store every iteration and select by visible evaluator score or
Pareto policy. Multi-aspect quality can regress on one axis while improving
another, so "last draft wins" is too weak for repo code.

## After Paper 011

Tree of Thoughts adds the first explicit search layer:

```text
state -> expand -> heuristic evaluate -> select -> backtrack
```

For `alpha_evolve`, the "thought" should not be private reasoning. It should be
an auditable artifact:

```text
patch plan
generated-test suite
prompt mutation
microlib draft
loss-shaper design
evaluator-stage proposal
```

The most important implementation lesson is state size. A useful state is small
enough to branch and large enough to score. For repo work, that argues for
branching over patch plans and microlib-sized artifacts before spending full
Codex edit calls.

ToT also sharpens the distinction between heuristic and fitness:

```text
heuristic = guides frontier search
visible evaluator = cheap deterministic feedback
hidden fitness = promotion gate
```

LLM value/vote scores can prune a current frontier, but they should never delete
archive records or promote code. The Mini Crosswords ablation is the warning:
pruning heuristics can reject states that are actually good. Pruned candidates
need archive entries and resurrection conditions.

Local microlibs added by ToT:

```text
thought_state_schema
candidate_expander
state_heuristic_evaluator
frontier_selector
backtracking_controller
prune_archive
```

This creates a practical order:

```text
linear repair for concrete failures
Self-Refine for soft artifacts
ToT over patch plans for ambiguous targets
LLaMEA serial mutation for tested candidate families
AlphaEvolve-style database/islands after frontier search beats equal-budget sampling
```

## After Paper 012

LATS turns ToT into action search with environment feedback:

```text
select -> expand -> evaluate -> simulate -> backpropagate -> reflect
```

This is the first strong blueprint for a Codex search runner over executable
candidate patches. The transfer is conditional: LATS only makes sense when
candidate states can be reverted or replayed. In this repo that means isolated
worktrees or patch sandboxes. A tree search that edits the main working tree in
place is invalid by construction.

The local reward split becomes:

```text
visible_reward:
    generated tests, unit tests, smoke tests, benchmark slices

hidden_fitness:
    repo-owned heldout gate and promotion decision

branch_reflection:
    failed trajectory diagnosis scoped to descendants
```

Backpropagate visible reward through the candidate tree. Promote only with
hidden/repo-owned gates. Feed failed reflections to descendants, not unrelated
siblings, unless the lesson is promoted to family/global memory by evidence.

Local microlibs added by LATS:

```text
mcts_node_store
uct_selector
sandboxed_expansion_runner
visible_reward_backpropagator
branch_reflection_memory
rollback_contract_checker
```

This adds a target-dependent algorithm choice:

```text
linear Reflexion:
    concrete failure with clear evaluator output

ToT:
    branch over plans before editing

LATS:
    branch over executable actions when rollback and visible reward are cheap

LLaMEA/AlphaEvolve:
    longer-running archive/population search
```

## After Paper 013

SWE-bench adds the evaluator realism layer that every search method depends on:

```text
issue/problem statement
base repo state
patch prediction
evaluation sandbox
fail-to-pass repair tests
pass-to-pass maintenance tests
resolved/partial/no report
```

The key local lesson is that search is meaningless without a replayable task
harness. The DynaWorld evolver needs SWE-style microbenchmarks before comparing
one-shot Codex, Reflexion, ToT, LATS, or AlphaEvolve variants.

SWE-bench also makes localization a first-class stage. Larger context can hurt,
and BM25 can miss all oracle files for many instances. The alpha_evolve runner
therefore needs:

```text
context_retriever
oracle_context_ablation
context_pruner
localization_score
```

before full patch evolution. If oracle context works and retrieved context
fails, mutation is not the bottleneck.

Local microlibs added by SWE-bench:

```text
repo_task_schema
patch_prediction_format
fail_pass_grader
context_retriever
evaluation_sandbox
task_instance_builder
```

The promotion rule should mirror SWE-bench:

```text
resolved = fail_to_pass all pass AND pass_to_pass all pass
```

Candidate-visible tests can shape search, but hidden fail-to-pass/pass-to-pass
gates decide promotion.

## After Paper 014

SWE-agent adds the agent-computer-interface layer around the SWE-bench task
contract:

```text
task statement
    -> bounded context/search/view interface
    -> compact edit interface
    -> explicit observation feedback
    -> guardrails
    -> history compression
    -> final patch submission
```

The key local lesson is that the interface is part of the optimizer. A Codex
evolver should not just call `codex exec` with a loose prompt and hope that
better prompts solve navigation, edit reliability, and stale context. It should
shape the model-facing task packet, patch contract, evaluator report, and
history digest deliberately.

Local microlibs added by SWE-agent:

```text
aci_contract
context_packet
search_summary
patch_guard
observation_digest
history_compactor
trajectory_store
budget_controller
failure_classifier
interface_ablation_runner
```

SWE-agent also gives a failure taxonomy that should drive next mutation:

```text
localization failure:
    rebuild context packet

failed edit:
    repair patch mechanics

incorrect implementation:
    semantic mutation

overly specific implementation:
    add counterexamples and hidden tests

cannot reproduce:
    improve task harness before more search

timeout / slow failure:
    kill branch and preserve reflection
```

The result reinforces a split already implied by SWE-bench:

```text
full logs:
    stored in archive

prompt context:
    compressed, recent, evaluator-relevant state only
```

The first local `alpha_evolve` runner should therefore be an interface ablation
system before it is a fancy search algorithm. Test whole-file prompt versus
bounded context, raw evaluator logs versus observation digests, no patch guard
versus syntax/import guard, and full history versus compact failure summary on a
small DynaWorld task suite.

## After Paper 015

Agentless adds the baseline discipline that should sit underneath any
autonomous Codex loop:

```text
localize
    -> sample patches
    -> validate
    -> rank
```

The paper's main local consequence is simple: before building a SWE-agent or
LATS-style multi-turn system, build a staged runner that does not let the model
choose future actions. If that runner is competitive, evolution should target
localization, patch diversity, generated tests, and rankers before autonomy.

Local microlibs added by Agentless:

```text
agentless_baseline_runner
repo_structure_summarizer
file_localizer
related_element_localizer
edit_location_sampler
context_window_builder
patch_sampler
patch_parser
generated_repro_test_builder
regression_selector
patch_ranker
benchmark_sanitizer
```

The staged runner gives stage-level metrics:

```text
file localization recall
edit-location recall
patch parse success
syntax guard success
regression pass count
generated-test plausibility
ranker chosen-vs-best gap
hidden promotion score
```

This turns `alpha_evolve` from one opaque "did Codex fix it?" loop into a
diagnosable pipeline. If oracle context solves a task but retrieved context
fails, improve the localizer. If the sample pool contains a hidden-pass patch
but the selected patch fails, improve the ranker. If no sampled patch works,
improve mutation/generation.

Agentless also adds a benchmark-sanitization requirement. Local tasks are not
valid evolver benchmarks if they leak the patch, lack enough information, have a
misleading accepted solution, or reward formatting instead of behavior.

The emerging implementation order is now:

```text
0. task suite and hidden gates
1. one-shot Codex baseline
2. Agentless staged sample-and-rank baseline
3. SWE-agent-style interactive escalation for measured failure classes
4. AlphaEvolve population search over prompts, context packets, patches, tests,
   rankers, and evaluators
```

## After Paper 016

OpenHands adds the platform primitives that become useful once the staged
baseline grows beyond one-off scripts:

```text
agent state
event stream
typed actions
typed observations
sandbox runtime
skill registry
delegation hook
benchmark registry
runner quality-control tests
```

The key local distinction is scope. Agentless says the first solver should be a
small staged baseline. OpenHands says that if we compare many solver variants,
we need reproducible infrastructure around those stages.

Local microlibs added by OpenHands:

```text
event_stream
action_schema
runtime_adapter
sandbox_workspace
skill_registry
evaluator_registry
benchmark_runner
cost_tracker
delegate_schema
runner_qc
```

This clarifies the `alpha_evolve` folder shape:

```text
tasks/
context/
localize/
patch/
validate/
rank/
archive/
events/
runtime/
evaluators/
qc/
```

OpenHands also adds a hard anti-scope rule:

```text
do not build first:
    GUI
    cloud
    browser
    multi-user state
    full agent marketplace
```

Those may be useful platform features, but they do not answer whether Codex
evolution improves DynaWorld microlibs. The useful first platform piece is a
typed event stream connecting `codex exec`, patch parsing, guard checks,
evaluator runs, ranking, failure classification, and archive writes.

## After Paper 017

The Codex/HumanEval paper adds the metric layer for all patch sampling:

```text
functional correctness:
    code is correct if it passes tests

pass@k:
    probability at least one of k samples passes

oracle best-in-sample:
    upper bound from hidden evaluator over sampled candidates

selected pass@1:
    what the runner actually submits

ranker gap:
    oracle best-in-sample - selected result
```

This changes how the first `codex exec` experiments should be reported. A run
that samples 20 patches and finds one hidden-pass patch is not successful unless
the ranker can pick it. Conversely, if the sample pool contains no hidden-pass
patches, the problem is generation/mutation or context, not ranker quality.

Local microlibs added by the Codex/HumanEval paper:

```text
functional_correctness
pass_at_k
sample_runner
sample_deduper
oracle_selector
ranker_gap
sandbox_policy
humaneval_like_tasks
```

This also sharpens the task-suite plan:

```text
start:
    tiny HumanEval-like microlib tasks with fast unit tests

then:
    SWE-style repo patch tasks

then:
    expensive trainer/renderer objectives
```

Generated code and generated tests are untrusted. Every candidate needs timeout,
runtime identity, and sandbox metadata. The runner should refuse to report
pass@k when tasks have fewer than k completed samples unless the subset is
explicitly labeled.

## After Paper 018

Program Synthesis with LLMs adds the concrete MBPP-like task-suite design:

```text
natural-language description
self-contained function or microlib
visible assert tests
hidden challenge tests
many samples per task
functional correctness scoring
prompt-example ablations
```

This paper makes task quality a first-class part of the evolver. If a task is
ambiguous, leaks the solution, or has weak visible tests, the runner may look
good while learning the wrong behavior. The first local task suite should be
hand-authored and challenge-tested.

Local microlibs added by Program Synthesis with LLMs:

```text
mbpp_like_task_schema
prompt_example_selector
challenge_test_registry
sample_reliability_metrics
semantic_failure_classifier
dialog_feedback_adapter
execution_truth_gate
```

This also adds two metrics beyond pass@k:

```text
coverage:
    any sample solves the task

reliability:
    fraction of samples that solve the task
```

A high-coverage low-reliability task is a ranker/evaluator problem. A
low-coverage task is a generation/context problem.

The immediate implementation order is now sharper:

```text
1. MBPP-like DynaWorld microlib suite
2. pass@k + ranker-gap harness
3. Agentless staged patch baseline
4. SWE-style repo patch suite
5. AlphaEvolve population search
```

## After Paper 019 - AlphaCode

AlphaCode adds the missing selection kernel between one-shot Codex patching and
full AlphaEvolve. Its useful mechanism is not the 41B model; it is the
separation between generation, public filtering, behavioral clustering, and a
small hidden-test submission budget.

The first local runner should copy this structure:

```text
k Codex candidates
visible candidate-facing tests
behavior probes for visible passers
clusters by observed output
n selected hidden-gate submissions
oracle best-of-k comparison
```

The metric vocabulary becomes:

```text
pass@k:
    did any generated candidate pass the hidden gate?

n@k:
    did the selected n candidates pass the hidden gate?

ranker_gap:
    oracle best-of-k success minus selected success

visible_false_positive_rate:
    visible passers that fail hidden gates
```

This reframes the next implementation milestone. The immediate goal is not a
beautiful program database. It is a measurable question:

```text
Can a cheap selector choose better than random from a small Codex sample pool?
```

AlphaCode also makes generated probes safer to reason about. Generated inputs
do not have to be perfect hidden tests; they can still be useful as behavioral
descriptors for clustering. That suggests a local split:

```text
generated probes:
    allowed to be noisy
    used for signatures and diversity

hidden gates:
    hand-owned and immutable
    used for promotion
```

Local microlibs added by AlphaCode:

```text
candidate_pool_store
codex_sample_runner
visible_test_filter
generated_probe_builder
behavior_signature
behavior_clusterer
budgeted_submission_selector
hidden_gate_runner
false_positive_auditor
ranker_gap_reporter
```

The first target suite should be competitive-programming-shaped DynaWorld
utilities: config normalization, metric aggregation, result-table parsing,
renderer capability selection, prompt context packing, and evaluator
fingerprinting. Avoid full trainers and renderer kernels until the small harness
can report `n@k` and ranker gap.

## After Paper 020 - CodeT

CodeT closes the first 20-paper pass by turning generated tests into a concrete
selection algorithm. AlphaCode says "cluster by behavior and submit diverse
representatives." CodeT says pure cluster size is not enough: many wrong
candidates can agree on a trivial behavior. A selector should score agreement
with both generated tests and sibling candidates.

The local selector now has a clear first implementation:

```text
candidate samples:
    Codex-generated patches

generated probes:
    Codex-generated tests or fixtures from the task spec

matrix:
    candidate x generated_probe pass/fail/time/error

consensus sets:
    candidates with the same generated-probe pass vector

score:
    sqrt(candidate_count) * generated_probe_pass_count

promotion:
    only after hidden repo gates
```

The crucial boundary is:

```text
generated tests:
    selector evidence
    can be noisy
    can be toxic
    should be audited

hidden gates:
    promotion truth
    should not be generated by the same loop and treated as final authority
```

Local microlibs added by CodeT:

```text
generated_test_prompt_builder
generated_test_postprocessor
candidate_test_matrix
consensus_set_builder
dual_agreement_scorer
test_toxicity_auditor
coverage_probe_reporter
selector_ablation_reporter
```

The first benchmark report should compare:

```text
random visible passer
visible-test count only
AlphaCode-style largest cluster
CodeT-style dual agreement
oracle best-of-k
```

This is the cleanest bridge from the paper pass into code. Before building a
full AlphaEvolve archive, build a small microlib harness that proves generated
tests reduce ranker gap without replacing hidden gates.

## Open Synthesis Questions

- When should a DynaWorld microlib evolve code blocks versus whole-file patches?
- Is island diversity enough, or do we need explicit behavioral descriptors per
  problem?
- Should failed candidates feed future prompts as natural-language reflections,
  structured metrics, or both?
- Can generated tests help build evaluators for mixed scheduler and benchmark
  contracts, or will they amplify shallow proxies?
- What is the smallest runner that beats one-shot `codex exec` on a real
  microlib?
- Which candidate-visible scores are shaping signals, and which repo-owned
  metrics are immutable fitness gates?
- What promotion gate is strict enough for a candidate to become a reusable
  skill rather than only an archived attempt?
- Which target problems are repair loops, and which genuinely need evolutionary
  search over multiple candidate families?
- What should invalidate a reflection: evaluator fingerprint, data fingerprint,
  parent-family drift, or observed negative utility?
- Should feedback generation and patch generation be separate Codex calls for
  inspectability, or combined when the evaluator feedback is already clear?
- What is the minimum `feedback_quality` score that justifies spending another
  Codex refinement attempt?
- What state unit makes ToT search useful here: patch plan, patch diff, prompt
  section, generated-test suite, or evaluator-stage proposal?
- What is the policy for resurrecting candidates pruned by weak heuristic votes?
- What rollback mechanism is cheap and robust enough for Codex-expanded child
  nodes: git worktree, patch replay, or copied temp tree?
- Should MCTS backpropagate generated-test reward, smoke reward, or a
  risk-adjusted visible score?
- What is the first local SWE-style task suite for DynaWorld microlibs?
- Which failures are localization failures versus mutation/search failures?
- Which parts of the Codex interface need wrappers, and which should remain raw
  Codex/tool behavior?
- What is the cheapest interface ablation that predicts later evolution quality?
- How much of the first win comes from staged localization/selection before any
  autonomous loop is introduced?
- What is the ranker gap between the best sampled patch and the patch selected
  by cheap visible signals?
- What is the minimum event schema that makes candidate failures replayable
  without becoming a full platform?
- Which runtime adapter should be first: temp directory, git worktree, Docker,
  or all behind the same interface?
- What is the first local HumanEval-like task suite for testing pass@k and
  ranker gap before repo-level patching?
- How many Codex samples are enough before evaluator cost beats diversity gain?
- Which local microlib tasks need hand-written challenge tests before they are
  allowed into the benchmark suite?
- Should prompt-example sets be selected manually, by similarity, by diversity,
  or evolved as part of the population?
- Does largest-cluster selection work for repo patches, or do common wrong
  patches dominate the biggest cluster?
- What probe quality is sufficient for behavioral clustering when probes are
  not trusted as hidden truth?
- Should hidden gates rotate across runs to reduce benchmark leakage, or stay
  fixed for comparability?
- Can generated tests for DynaWorld microlibs be audited for toxicity without a
  full reference implementation?
- Should generated probes be built before candidate generation, after candidate
  generation, or both?
- What weighting between candidate support and test support is robust for repo
  patches: linear, square root, log, or hard cap?
