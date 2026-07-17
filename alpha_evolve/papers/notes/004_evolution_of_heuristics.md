# 004 - Evolution of Heuristics: Towards Efficient Automatic Algorithm Design Using Large Language Model

Status:
    first-pass

Primary sources:

- arXiv page, v3 current as of this read: https://arxiv.org/abs/2401.02051
- PDF: https://arxiv.org/pdf/2401.02051
- Official repository: https://github.com/FeiLiu36/EoH

Why this paper matters for `alpha_evolve`:

Evolution of Heuristics (EoH) is the strongest paper so far for designing the
operator taxonomy of a local Codex evolver. It is narrower than CodeEvolve:
the evolved object is a heuristic function or guided-local-search strategy. But
it adds one crucial thing FunSearch did not: each individual has both a
natural-language "thought" and executable code. That maps cleanly onto
DynaWorld microlibs where a candidate patch should carry a short strategy
summary, not only a diff and metrics.

One-sentence mechanism:

EoH evolves a population of heuristic thought-code pairs using five LLM prompt
operators, evaluates each executable heuristic on task instances, and selects
the best individuals for the next generation.

## Reading Questions

- What is the executable feedback signal?
- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
- What is the population/database/selection mechanism?
- What evidence proves the loop improves over one-shot generation?
- What does the method assume that DynaWorld does not have?

## Mechanism

EoH models an individual as:

```text
heuristic = {
  thought: natural-language description of the idea,
  code: executable function in a predefined format,
  fitness: score from running the heuristic on problem instances
}
```

The core loop:

```text
initialize N heuristics with LLM calls
for each generation:
    for each of five prompt strategies:
        call the strategy N times
        select parent heuristic(s)
        ask LLM for a new thought and code
        evaluate executable code on instances
        add feasible candidates to current population
    select the top N heuristics for the next generation
```

Important implementation details:

- Every generated heuristic is required to produce both a description and a
  code implementation.
- Code must follow a predefined function name/input/output format so the
  evaluator can load it mechanically.
- EoH evaluates each heuristic on a set of problem instances, not one input.
- Each generation can add up to `5N` new heuristics because there are five
  strategies, each called `N` times.
- E1/E2 use `p = 5` parent heuristics in the paper's experiments.
- Parent selection is rank-biased: if `r_i` is the rank and `N` is population
  size, selection probability is proportional to `1 / (r_i + N)`.
- The paper uses 20 generations for all three main problems.
- Population size is 20 for online bin packing and 10 for TSP/FSSP.
- The main experiments use GPT-3.5-turbo and run on a single i7-9700 CPU.

This is deliberately simpler than CodeEvolve's islands, MAP-Elites, migration,
and prompt populations. Its value is the operator vocabulary and the explicit
thought-code representation.

## Prompt Operators

EoH uses one initialization prompt and five evolution prompt strategies.

### Initialization

The LLM receives the task description and output schema, then designs a new
heuristic from scratch:

```text
describe heuristic in one sentence
then implement Python function with fixed name, inputs, and outputs
```

No parent heuristic is included.

### E1 - Explore Different

The LLM receives several parent heuristics and is instructed to design a new
heuristic that is as different as possible from them. This is explicit novelty
pressure.

Local translation:

```text
Given these accepted candidates and their failure modes, propose a substantially
different implementation strategy while preserving the microlib contract.
```

### E2 - Extract Common Idea Then Vary

The LLM receives several parent heuristics, first identifies their common idea,
then designs a new heuristic based on that backbone while adding new parts. In
the appendix, the E2 prompt for bin packing asks the model to:

```text
identify common idea
describe new heuristic
implement fixed Python function
```

This is the most useful EoH operator for DynaWorld. It is a structured way to
make Codex synthesize across multiple near misses without dumping the whole
history.

### M1 - Modify For Better Performance

The LLM receives one parent heuristic and is asked to modify it to produce a
better heuristic. This is local improvement.

Local translation:

```text
Take this accepted candidate and improve the primary metric without changing
the stage-0 and stage-1 contract.
```

### M2 - Parameter Retune

The LLM receives one heuristic and is asked to try different parameters inside
the current heuristic instead of designing a new one.

Local translation:

```text
Retune thresholds, constants, schedules, or scoring weights. Do not change the
interface or architecture.
```

This is directly useful for `gaussian_512_promotion_guard` and scheduler
policy functions.

### M3 - Simplify

The LLM receives one heuristic and is asked to identify redundant components,
then simplify the code.

Local translation:

```text
Reduce changed LOC, remove unnecessary branches, keep evaluator metrics within
tolerance, and preserve hard invariants.
```

This is important because evolutionary code search tends to accrete hacks.
Without a simplification operator, the archive may become full of brittle
overfit patches.

## Evaluation

The paper evaluates EoH on three combinatorial optimization problems.

### Online Bin Packing

Task:

- Incoming items must be assigned online to bins with fixed capacity.
- EoH evolves a scoring function:

```text
score(item, bins) -> scores_per_feasible_bin
```

Evaluation:

- The evolution fitness uses five Weibull instances with 5k items and capacity
  100.
- Reported test sets range from 1k to 10k items and capacities 100 and 500.
- Metric is fraction of excess bins to lower bound, lower is better.

Main table:

```text
First Fit: 5.32, 4.40, 4.44, 4.97, 4.27, 4.28
Best Fit:  4.87, 4.08, 4.09, 4.50, 3.91, 3.95
FunSearch: 3.78, 0.80, 0.33, 6.75, 1.47, 0.74
EoH:       2.24, 0.80, 0.61, 2.13, 0.78, 0.61
```

Interpretation:

- EoH beats human first-fit/best-fit across the table.
- EoH matches FunSearch on 5k/C100, beats it on most listed settings, and loses
  to FunSearch on 10k/C100.
- EoH reaches this with a few thousand LLM queries, while the paper cites
  FunSearch as around one million queries on this problem.
- The bin-packing evolution improves from fitness 0.962 to 0.993 over 20
  generations with 2,000 LLM queries.

The important local lesson is not that EoH always dominates FunSearch. It is
that thought-code operators can reach useful heuristics on a small budget.

### Traveling Salesman Problem

EoH designs a guided local search (GLS) heuristic. Instead of directly
constructing a full route, it designs a strategy to update the distance matrix
when local search is stuck.

Evolved function shape:

```text
update_edge_distance(edge_distance, local_opt_tour, edge_n_used)
    -> updated_edge_distance
```

Evaluation:

- Evolution uses 64 random TSP100 instances sampled from `[0, 1]^2`.
- Fitness is average gap to the optimal solution from Concorde.
- Local search uses relocate and 2-opt operators.
- Test reporting includes TSPLib instances.

Reported main-table examples:

- EoH reaches best-known solutions on pr124, kroA150, and u159.
- EoH is competitive with or better than OR-Tools and neural solvers on the
  listed TSPLib subset under the paper's settings.

Local lesson: the evolved object can be a landscape-update policy inside a
fixed solver, not the solver itself. This maps to DynaWorld schedules, guards,
and loss/metric shaping better than full trainer synthesis.

### Flow Shop Scheduling

EoH again designs a GLS perturbation/update heuristic, this time for
permutation flow-shop scheduling.

Evolved function shape:

```text
heuristic(current_sequence, time_matrix, m, n)
    -> new_time_matrix, perturb_jobs
```

Evaluation:

- Evolution uses 64 random instances with 50 jobs and 2-20 machines.
- Test reporting uses Taillard instance sets.
- The paper reports EoH outperforming human heuristics and recent neural
  solvers on the main listed sets.

Local lesson: for DynaWorld, an evolved policy can return both a transformed
state and an action subset. That is relevant to promotion guards, scheduler
batch choices, and candidate filtering.

## Ablations

The paper ablates thought-code representation and prompt strategies on online
bin packing.

Variants:

- `EoC`: code-only variant using E1.
- `EoH-e1`: thought + code but only E1.
- `EoH-e2`: thought + code with E1 and E2.
- `EoH`: thought + code with E1, E2, M1, M2, M3.

The main ablation table shows:

- EoH is best on average.
- EoH-e2 is usually second best.
- EoC performs worst or second worst.

The separate thought/code study compares:

- `C2C`: only code representation.
- `T2T2C`: only thought in evolutionary prompts, then code is generated for
  evaluation.
- `T&C2T2C`: thought and code as input, but output only thought before a later
  code-generation step.
- EoH: thought and code as both input and output.

Reported averages on 5k Weibull bin-packing instances:

```text
C2C:      2.57
T2T2C:    2.13
T&C2T2C:  0.85
EoH:      0.66
```

This is the key result for `alpha_evolve`: the thought is not just prose for
humans, and the code is not enough by itself. The combination gives the model
semantic compression plus executable grounding.

The paper also compares LLMs:

```text
Sampling GPT3.5 10k queries: average 2.44
EoH CodeLlama 2k queries:    average 1.07
EoH Deepseek 2k queries:     average 1.41
EoH Gemini Pro 2k queries:   average 0.71
EoH GPT3.5 2k queries:       average 0.66
```

Interpretation: operator structure matters enough that weaker or cheaper
models with EoH can beat more brute-force sampling, though stronger models
still help.

Finally, the expert-seed experiment inserts the FunSearch heuristic into the
initial population. EoH with that expert seed improves further:

```text
FunSearch average: 0.97
EoH average:       0.66
EoH expert:        0.55
```

Local lesson: seeding the population with a known good hand patch or current
repo baseline is useful. The runner should not start from blank prompts when a
baseline implementation exists.

## Why It Beats One-Shot Codex

EoH beats one-shot generation because it supplies a low-budget loop with typed
variation operators:

1. The LLM must emit both strategy and executable code.
2. Multiple prompt strategies create different search pressures.
3. E1/E2 do crossover-like exploration over parent heuristics.
4. M1/M2/M3 do local refinement, retuning, and simplification.
5. Evaluators select the next population by measured fitness.
6. The thought carries a compressed hypothesis that future prompts can reuse.

The analog for local Codex is that repeated generic prompts are too blunt. The
runner should schedule candidate operators:

```text
explore_different
extract_common_backbone
improve_parent
retune_constants
simplify_patch
```

and log the operator that produced each candidate.

## DynaWorld Mapping

### Candidate Record Update

After EoH, each `programs.jsonl` row should include a candidate thought:

```json
{
  "candidate_id": "cand_000019",
  "operator": "retune_constants",
  "thought": "Delay 512px promotion until finite render/loss diagnostics pass, then checkpoint immediately before switching.",
  "code_or_patch_path": "candidates/cand_000019/patch.diff",
  "metrics": {
    "promotion_reached": true,
    "finite": true,
    "checkpoint_before_promotion": true
  }
}
```

This thought is not trusted as truth. It is prompt material and a debugging
index. The evaluator remains authoritative.

### Prompt Contract Update

The current `alpha_evolve/prompt_contract.md` should eventually support
operator-specific prompt sections:

```text
Operator:
  extract_common_backbone

Parent candidates:
  cand_000011: thought, changed files, metrics, failure note
  cand_000014: thought, changed files, metrics, failure note
  cand_000017: thought, changed files, metrics, failure note

Task:
  First identify the common implementation idea.
  Then propose one new thought.
  Then make one patch implementing that thought.
```

The final response schema should request:

```text
- candidate thought
- changed files
- implementation summary
- tests run
- expected evaluator risks
```

### Operator Schedule

For a first local serial runner:

```text
population_size = 8
operators per generation:
  2 explore_different
  2 extract_common_backbone
  2 improve_parent
  1 retune_constants
  1 simplify_patch
select top 8 by hard-gated score plus diversity descriptors
```

This is much smaller than EoH and intentionally serial-friendly. It preserves
operator diversity without requiring islands yet.

### Microlib Fit

Best fit: `gaussian_512_promotion_guard`

- It has natural thought-code/patch candidates:
  "checkpoint before promotion", "abort before optimizer step", "diagnostic
  source classifier", "promotion schedule retune".
- M2 can retune thresholds and timing.
- M3 can simplify guard code after a working candidate appears.

Best fit: `mixed_same_view_novel_scheduler`

- EoH's policy-function framing maps to:

```text
choose_batch_kind(state) -> same_view_recon | heldout_view_recon
```

- E1 can explore different curriculum schedules.
- E2 can merge several partial policies.
- M2 can retune ratios and warmup.

Good fit: `code_org_helpers`

- M3 is valuable here. Simplification should be a first-class operator, not a
  cleanup afterthought.
- The thought helps preserve intent: "deduplicate RGB composition while keeping
  F3 and F32 behavior identical."

Weak initial fit: `star_uvt_feature_rgb_handoff`

- The paper's success depends on fixed function schemas and short evaluations.
- STAR UVT feature handoff spans too many surfaces unless reduced to a small
  policy/helper subproblem.

### Expert Seeds

EoH's expert-seed result means local evolution should include current baselines
as seed candidates:

- current hand-written helper
- current benchmark-positive prototype
- best failed near miss with clear failure note
- current simplest passing implementation

Do not hide the existing solution from Codex. The goal is not blank-slate
creativity; it is measured improvement over a known baseline.

## Failure Modes

### Thought Can Become Rationalization

The model may write a plausible thought that does not match the patch. Local
runner should extract both:

```text
declared_thought
actual_diff_summary
evaluator_failure_reason
```

If the thought and diff disagree, future prompts should prefer the evaluator
failure reason over the declared thought.

### Fitness Overfit

EoH's appendix notes that code-only EoC can overfit the training distribution.
DynaWorld is more vulnerable because smokes are often tiny. Every evolved
heuristic/policy must be checked against at least one heldout or varied
evaluator input before promotion.

### Prompt Operators Can Dilute Hard Constraints

E1 asks for difference. In a repo, "different" can mean "different because it
breaks the contract." The prompt must say:

```text
Be different in implementation strategy, not in target, allowed paths, inputs,
outputs, loss kind, frame count, data semantics, or evaluator commands.
```

### Retuning Can Hide Downgrades

M2 is useful but dangerous. It can win by lowering resolution, shortening
schedule, or reducing work. Retune prompts must explicitly list immutable
config fields.

### Simplification Can Remove Diagnostics

M3 can delete code that looks redundant but protects a failure mode. The
simplification evaluator must include the historical failure case.

## Falsification Tests

### Test 1 - Thought Plus Patch Versus Patch Only

Run a cheap microlib twice:

```text
A: prompts include prior patch diffs and metrics only
B: prompts include prior thoughts, patch diffs, and metrics
```

Support for EoH transfer:

- B produces fewer repeated failures
- B creates more distinct accepted strategies
- B reaches same evaluator stage with fewer candidates

Falsification:

- thoughts add prompt bloat and do not improve candidate quality

### Test 2 - Operator Mix Versus Generic Improve

Compare:

```text
A: all candidates use "improve current best"
B: scheduled E1/E2/M1/M2/M3-style operators
```

Support:

- B has better accepted diversity or final score under equal candidate budget

Falsification:

- generic improve performs equally and is simpler

### Test 3 - Simplification Operator

After a valid but messy candidate appears, run M3-style simplification:

```text
input: candidate thought, diff, metrics, historical failure case
task: reduce LOC while preserving all hard gates
```

Support:

- simplified patch keeps stage pass depth and reduces changed LOC

Falsification:

- simplification repeatedly removes diagnostics or breaks hidden invariants

### Test 4 - Expert Seed

Seed one run with the current hand baseline and one run from a blank/trivial
implementation.

Support:

- expert-seeded run reaches useful candidates faster

Falsification:

- expert seed anchors the search too strongly and hurts diversity

## Design Decisions For `alpha_evolve`

1. Add `thought` as a first-class candidate field.
2. Add `operator` as a first-class candidate field.
3. Support EoH-style operator prompts:
   `explore_different`, `extract_common_backbone`, `improve_parent`,
   `retune_constants`, and `simplify_patch`.
4. Keep thoughts short, one to three sentences, and require code/patch evidence.
5. Use current repo baselines as expert seeds.
6. Require fixed function/patch interfaces for early microlib proof runs.
7. Treat M3 simplification as a performance feature, because code bloat is a
   real evolutionary failure mode.
8. Always include immutable hard constraints in E1/E2 prompts.
9. Evaluate candidates on varied inputs before promotion because EoH's own
   code-only variant shows overfit risk.
10. Do not promote thought-only gains. No patch and no evaluator improvement
    means rejected candidate.

## Notes For Future Papers

- LLaMEA should be read next because it pushes the evolved object up one level:
  generated metaheuristics rather than one heuristic component.
- Eureka is the closest next analog for DynaWorld evaluators, because it evolves
  reward code and uses feedback from simulation.
- Later agent-loop papers should be read through the EoH lens: do they provide
  useful operator types, or just a generic "reflect and retry" loop?
