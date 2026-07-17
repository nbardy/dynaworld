# 003 - CodeEvolve: an open source evolutionary coding agent for algorithmic discovery and optimization

Status:
    first-pass

Primary sources:

- arXiv page, v4 current as of this read: https://arxiv.org/abs/2510.14150
- PDF: https://arxiv.org/pdf/2510.14150
- Framework repository: https://github.com/inter-co/science-codeevolve
- Experiment artifacts repository: https://github.com/inter-co/science-codeevolve-experiments

Why this paper matters for `alpha_evolve`:

CodeEvolve is the bridge between FunSearch's narrow function evolution and
AlphaEvolve's broader code-editing loop. It is open source, uses SEARCH/REPLACE
diffs, keeps island populations, supports prompt evolution, and exposes enough
engineering details to inform a local `codex exec` runner for DynaWorld.

One-sentence mechanism:

CodeEvolve evolves complete candidate programs with an islands-based genetic
algorithm, LLM-generated structured diffs, inspiration-based semantic
crossover, prompt meta-evolution, depth-limited exploitation through ancestor
history, sandboxed evaluation, migration, and optional MAP-Elites archives.

## Reading Questions

- What is the executable feedback signal?
- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
- What is the population/database/selection mechanism?
- What evidence proves the loop improves over one-shot generation?
- What does the method assume that DynaWorld does not have?

## Mechanism

CodeEvolve defines a solution as a candidate program and a prompt as the text
used to generate a solution. It tracks both:

```text
solution population S_i^t for island i at epoch t
prompt population P_i^t for island i at epoch t
solution fitness f_sol(S)
prompt fitness f_prompt(P) = max fitness of solutions produced by P
metric vector h(S) = runtime, memory, objective value, etc.
```

The practical loop is:

```text
initialize islands with seed solution(s) and initial prompt(s)
for each island and epoch:
    choose exploration or exploitation
    sample parent solution, inspirations, and maybe prompt
    ask LLM ensemble for SEARCH/REPLACE diffs
    apply diff to candidate code
    evaluate in sandbox with time/memory limits
    store metrics, logs, lineage, prompt, and fitness
    update island population and optional MAP-Elites archive
periodically migrate elites between islands
```

### Depth Exploitation

Depth exploitation refines high-performing solutions. The parent is sampled by
rank-based selection, so better solutions are more likely to be chosen. The LLM
sees:

```text
parent prompt P(S)
target solution S
k nearest ancestors A_k(S)
inspiration solutions I
```

This is an explicit answer to a local failure mode: if prompts only show the
current best patch, the model may rewrite the whole approach or forget why the
patch works. A bounded ancestor depth gives the model incremental context
without dumping the whole run history.

### Meta-Prompting Exploration

Exploration samples a solution and a prompt independently, then asks an
auxiliary LLM to enrich the prompt by analyzing the previous prompt and
solution. The main LLM then generates a new solution from the enriched prompt,
with no ancestor chain. This is meant to produce new search trajectories rather
than small local edits.

For DynaWorld, this should be treated carefully. Prompt evolution can help when
we are stuck in a local strategy, but it can also mutate away hard repo
constraints. Any local meta-prompting step must preserve immutable sections:
allowed paths, forbidden changes, evaluator commands, and final response
schema.

### Inspiration-Based Crossover

Both exploration and exploitation receive inspiration solutions. In
exploitation these are rank-sampled; in exploration they are sampled randomly.
Instead of splicing code directly, the LLM performs semantic crossover by
reading multiple successful solutions and generating a coherent diff.

This generalizes FunSearch's `k = 2` best-shot prompting. The local equivalent
is not just "include a winner." A prompt should include:

```text
target parent patch
1-3 inspiration patches with metrics
clear instruction to synthesize one coherent patch
```

### Structured Diffs

The public repository README says CodeEvolve uses SEARCH/REPLACE diffs and only
modifies specified code blocks between markers. This is more directly useful
for `codex exec` than whole-file generation:

```text
<<< SEARCH
old code
= = = = = = =
new code
>>> REPLACE
```

Local DynaWorld implication: if we build a runner, we should decide per
microlib whether Codex edits files directly and we harvest `git diff`, or
whether we force a SEARCH/REPLACE patch format. Direct edits fit Codex today;
SEARCH/REPLACE is easier to validate, replay, and constrain.

### Population Management

CodeEvolve adds successful candidates to the island population and stores
failures with logs. It migrates top performers to neighboring islands at a
fixed migration frequency/rate. Two anti-collapse details matter:

- A migrated solution can migrate at most once from its origin island.
- The best-performing solution of an island is not migrated, preserving
  island-specific uniqueness.

The paper also supports per-island MAP-Elites archives. Users define feature
descriptors, and each archive cell stores the highest-fitness solution for that
behavioral niche. The paper reports that MAP-Elites, especially CVT-MAP-Elites,
is important for surpassing AlphaEvolve in its CirclePackingSquare ablation.

For DynaWorld, MAP-Elites descriptors should be repo-specific, not generic:

```text
changed_loc_bucket
stage_pass_depth
batch_kind_coverage
gradient_presence
overflow_bucket
timing_bucket
loss_bucket
leakage_status
```

## Evaluation

The paper evaluates on benchmark suites used for AlphaEvolve and Evolution of
Heuristics.

Primary AlphaEvolve-style tasks:

- CirclePackingSquare with n = 26, 32.
- CirclePackingRect with n = 21.
- HexagonPacking with n = 11, 12.
- MinimizeMaxMinDist with n = 16, d = 2 and n = 14, d = 3.
- FirstAutocorrIneq and SecondAutocorrIneq.

Additional EoH tasks:

- Online bin packing.
- Traveling salesman.
- Flow shop scheduling.

Experimental setup:

- Runs use AWS SageMaker.
- Candidate programs execute in isolated sandboxes.
- Runtime and memory limits are enforced per candidate.
- Two main model configurations are compared: Gemini 2.5 and Qwen3-Coder-30B.
- The paper compares against reported AlphaEvolve numbers, reported
  ThetaEvolve numbers, and reruns of OpenEvolve and ShinkaEvolve under matched
  settings.

Main results reported by the paper:

- CodeEvolve matches or surpasses AlphaEvolve on 5 of 9 benchmark instances.
- It reports new best-known results on MinimizeMaxMinDist and
  CirclePackingSquare(n = 32).
- Qwen3-Coder-30B gives the strongest results on CirclePackingSquare, while
  Gemini 2.5 performs best on CirclePackingRect and the MinimizeMaxMinDist
  instances.
- In CirclePackingSquare(n = 26), Qwen3-Coder-30B surpasses AlphaEvolve after
  roughly 900 model calls at about 6 USD of API cost; Gemini 2.5 needs roughly
  400 calls at just under 35 USD.

The last point matters for local planning: a cheaper model with good
orchestration can beat a stronger model with weaker search economics. For
DynaWorld, the expensive part may be evaluator time, not LLM time, so the same
logic becomes:

```text
optimize candidate throughput before buying larger model calls
keep stage-0/stage-1 evaluators very cheap
reserve trainer/kernel smokes for filtered candidates
```

## Ablations

The central ablation uses CirclePackingSquare(n = 32) with Qwen3-Coder-30B.

The paper compares:

- Full method.
- Naive evolution: standard exploration/exploitation without the proposed
  components.
- No evolution: repeated prompts from the initial prompt and solution, with no
  contextual data from other candidates.

Findings:

- Full method has better mean performance and sample efficiency.
- For n = 32, full method is the only configuration that surpasses
  AlphaEvolve.
- For n = 26, the naive baseline needs more than twice as many evaluations to
  match the full method.

Depth/inspiration ablation:

- Depth-only configurations do not exceed AlphaEvolve.
- Inspiration-only with 2 or 3 inspirations does exceed AlphaEvolve.
- Full method outperforms both, suggesting synergy between ancestor context and
  inspiration crossover.

MAP-Elites/migration ablation:

- MAP-Elites is reported as necessary for surpassing AlphaEvolve in the
  CirclePackingSquare(n = 32) ablation.
- CVT-MAP-Elites performs best among the tested elite-selection policies.
- Cycle migration topology is the only tested topology that surpasses
  AlphaEvolve; Complete migration is only slightly better than no migration and
  appears to reduce diversity.

Practical defaults from the paper:

- Expensive closed-source ensemble: about 5 islands and 200 epochs.
- Open-weight model: about 10 islands and 250 epochs.
- Inspirations: 2-3.
- Ancestor depth: 3-5.
- Migration rate: 0.1.
- Initial exploration rate: 0.2 with plateau scheduling.
- Ring/cycle topology.
- CVT-MAP-Elites with fitness and evaluation time as feature descriptors.

For DynaWorld, do not copy these numbers blindly. The useful transfer is the
shape:

```text
small number of islands
low but nonzero exploration
bounded ancestor depth
2-3 inspirations
cycle-style migration
behavioral archive keyed by meaningful descriptors
```

## Why It Beats One-Shot Codex

CodeEvolve's evidence is stronger than "sampling more." It isolates several
ways that history changes the search:

1. Exploitation sees a parent and short lineage, which supports incremental
   repair instead of restarts.
2. Inspiration solutions provide semantic crossover.
3. Prompt evolution creates new exploration directions.
4. MAP-Elites keeps diverse behavior alive.
5. Island migration spreads useful ideas without collapsing all islands into
   one population.
6. Sandboxed evaluation turns patches into metrics and failure logs.

The local equivalent of the paper's "no evolution" baseline is:

```text
for i in range(N):
    codex exec same_prompt
    run same evaluator
```

This is not enough for the requested local `alpha_evolve` design. A real runner
needs parent IDs, prompt IDs, inspiration IDs, score signatures, and failure
logs. Otherwise it cannot reproduce the paper's gains.

## DynaWorld Mapping

### Runner Design Changes

The existing `alpha_evolve/codex_evolver_design.md` already has candidate
worktrees, program DB rows, islands, staged evaluators, and prompt sampling.
CodeEvolve suggests adding these fields to candidate state:

```json
{
  "candidate_id": "cand_000123",
  "prompt_id": "prompt_000017",
  "parent_id": "cand_000087",
  "ancestor_ids": ["cand_000070", "cand_000081", "cand_000087"],
  "inspiration_ids": ["cand_000025", "cand_000099"],
  "operator": "depth_exploit",
  "island": "cycle_02",
  "migration_origin": null,
  "archive_cell": "finite_grad_lowloc_fast",
  "fitness": 0.72,
  "metrics": {
    "correct": true,
    "stage_pass_depth": 2,
    "changed_loc": 44,
    "eval_seconds": 18.2
  },
  "failure_log_path": null
}
```

And these prompt-row fields:

```json
{
  "prompt_id": "prompt_000017",
  "parent_prompt_id": "prompt_000010",
  "operator": "meta_prompt",
  "immutable_contract_hash": "sha256...",
  "rendered_prompt_path": "prompts/prompt_000017.md",
  "best_child_fitness": 0.72
}
```

The immutable contract hash is the local guard against meta-prompt drift.
Prompt evolution may rewrite strategy language, but not allowed paths,
forbidden edits, evaluator commands, or final response requirements.

### Direct Codex Edits Versus SEARCH/REPLACE

CodeEvolve's SEARCH/REPLACE format is attractive because it is replayable and
scope-checkable. Codex already edits files directly well. The local runner can
support both modes:

```text
mode=direct_edit:
    codex exec edits candidate worktree
    runner records git diff

mode=structured_patch:
    codex exec writes SEARCH/REPLACE blocks
    runner applies them, then records git diff
```

Use `direct_edit` first for DynaWorld because it matches current Codex
behavior. Use `structured_patch` for high-risk microlibs where replayability
and patch-zone restriction matter more than agent ergonomics.

### Microlib Fit

Best fit: `code_org_helpers`

- CodeEvolve's patch evolution maps well to small behavior-helper refactors.
- Evaluator can combine focused tests, changed LOC, and smoke if call
  signatures changed.
- MAP-Elites descriptors can preserve different helper shapes rather than only
  shortest diff.

Strong fit: `gaussian_512_promotion_guard`

- Depth exploitation can refine a guard patch through staged smoke failures.
- Failure logs are especially useful: first nonfinite source, checkpoint
  presence, promotion reached, optimizer corruption avoided.
- Inspirations might combine a diagnostic-heavy patch with a minimal
  checkpoint-before-promotion patch.

Moderate fit: `mixed_same_view_novel_scheduler`

- CodeEvolve can evolve bounded trainer/config patches.
- Meta-prompting is risky because it may mutate away data-contract language.
- The immutable prompt contract must preserve same-view versus heldout
  semantics.

Risky fit: `star_uvt_feature_rgb_handoff`

- The real implementation spans Python trainer code, Metal kernels, extension
  build behavior, parity, gradients, and timing.
- CodeEvolve-style depth exploitation is useful after a valid seed exists.
- It is a bad first proof target unless the evolved surface is narrowed to an
  experiment-side helper or prototype.

### Selection And Migration

For local `codex exec` evolution, use a tiny version:

```text
islands = 3
topology = cycle
migration_every = 10 accepted candidates
migration_rate = 1 candidate per migration event
do not migrate the island's current best
ancestor_depth = 3
inspirations = 2
exploration_rate = 0.2
```

This is not claimed as optimal. It is the smallest version that preserves the
paper's core mechanism without overbuilding.

### Evaluator Cost Discipline

CodeEvolve's benchmark tasks often have short sandbox evaluations. DynaWorld
evaluators can become expensive quickly. The local system needs explicit
budget tiers:

```text
stage0 static scope and patch validity: seconds
stage1 unit/parity JSON: seconds to low minutes
stage2 smoke/timing: low minutes
stage3 benchmark/train comparison: rare elite only
```

The prompt sampler should include each candidate's highest stage passed. A
stage-1 pass with a promising design can be a valid inspiration even if it
failed stage 2.

## Failure Modes

### Prompt Evolution Can Break Contracts

CodeEvolve evolves prompts. DynaWorld cannot let evolved prompts rewrite safety
and repo constraints. A prompt row needs separate mutable and immutable
sections:

```text
immutable:
    problem, allowed paths, forbidden paths, evaluator commands, hard rejects
mutable:
    search strategy, hypotheses, inspiration summary, failure lessons
```

Reject a generated prompt if the immutable hash changes.

### MAP-Elites Needs Real Descriptors

Using fitness and evaluation time as descriptors may be enough for generic
algorithmic benchmarks. It is too weak for DynaWorld. A fast invalid gradient
hack and a legitimate optimization can occupy similar runtime cells. The
descriptor must encode the behavior we care about.

### Complete Migration Can Collapse Diversity

The paper's topology ablation warns against over-sharing. For local use, do not
make every candidate visible to every prompt by default. Prompts should sample
from one island plus a small number of global inspirations.

### Reported Comparisons Are Partly Indirect

AlphaEvolve is closed-source, so the paper compares against reported
AlphaEvolve values rather than rerunning it. That is acceptable for reading the
system design, but local DynaWorld claims must be replayed in our own worktree.
Do not cite a paper-style comparison as evidence that a local runner works.

### The Framework Assumes Easy Sandboxing

CodeEvolve's repo says it was developed/tested on Linux, and notes Linux-only
CPU pinning and per-island CPU partitioning. DynaWorld runs often happen on
macOS/MPS. Candidate isolation must not depend on Linux CPU affinity. For local
Mac proof, use process-level timeouts and worktree isolation first; add CPU
pinning only for Linux/cloud runners.

## Falsification Tests

### Test 1 - Direct Edit Versus Structured Patch

Run the same low-risk `code_org_helpers` microlib in two modes:

```text
A: Codex edits files directly; runner records git diff
B: Codex emits SEARCH/REPLACE blocks; runner applies them
```

Measure:

- accepted candidate rate
- patch application failures
- evaluator pass depth
- changed LOC
- repeated reject patterns

Support for structured patches:

- similar accepted rate with fewer scope violations and easier replay

Support for direct edits:

- much higher accepted rate with no meaningful increase in scope violations

### Test 2 - Inspirations Add Value

For a cheap microlib, run:

```text
0 inspirations
1 inspiration
2 inspirations
3 inspirations
```

Keep evaluator budget fixed.

Expected if CodeEvolve transfers:

- 2 inspirations should beat 0-1 on diversity or final score.
- 3 may help but risks prompt bloat.

### Test 3 - Prompt Evolution With Immutable Hash

Allow meta-prompting to rewrite only mutable search strategy sections. Reject
if immutable sections change.

Support:

- evolved prompts reduce repeated failure modes without changing allowed paths
  or hard rejects

Falsification:

- prompt evolution mostly creates invalid prompts or dilutes constraints

### Test 4 - Cycle Migration Versus Shared Global Pool

Compare:

```text
A: three islands with cycle migration
B: one global pool visible to all prompts
C: three islands with no migration
```

Support:

- cycle migration produces more accepted diversity and better final score than
  global pool or no migration

Falsification:

- global pool is simpler and just as good for small local microlibs

## Design Decisions For `alpha_evolve`

1. Keep FunSearch-style function evolution as the first proof mode.
2. Add CodeEvolve-style patch evolution as the second mode, using direct Codex
   edits first and optional SEARCH/REPLACE later.
3. Add prompt IDs and prompt fitness to the candidate database.
4. Store parent, ancestor, inspiration, operator, island, migration origin, and
   archive-cell metadata.
5. Use immutable/mutable prompt sections before allowing meta-prompting.
6. Start with 3 islands and cycle migration, not a large distributed system.
7. Prefer behavior descriptors over generic runtime-only MAP-Elites cells.
8. Include failure logs as first-class prompt material, but keep them short.
9. Treat evaluator-stage pass depth as a useful selection metric.
10. Do not claim local success until the archive loop beats repeated one-shot
    `codex exec` under equal candidate budget.

## Notes For Future Papers

- Evolution of Heuristics should be read next because CodeEvolve depends on the
  heuristic-design lineage and uses EoH benchmarks.
- LLaMEA should clarify how much of the population/search machinery is useful
  when the evolved object is a metaheuristic rather than a concrete program.
- Agentless later needs to challenge whether all this machinery beats a simple
  localization-plus-repair loop on actual repo bugs.
