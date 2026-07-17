# 002 - Mathematical discoveries from program search with large language models

Status:
    first-pass

Primary sources:

- Nature page: https://www.nature.com/articles/s41586-023-06924-6
- Author PDF: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/funsearch-making-new-discoveries-in-mathematical-sciences-using-large-language-models/Mathematical-discoveries-from-program-search-with-large-language-models.pdf
- Official code/data repository: https://github.com/google-deepmind/funsearch

Why this paper matters for `alpha_evolve`:

FunSearch is the clean predecessor to AlphaEvolve. It strips the problem down
to the essential loop: evolve a small function inside a known skeleton, execute
it against a hard evaluator, store only correct programs, then prompt the LLM
with measured high performers. For DynaWorld, this paper is the strongest
argument that the first local runner should evolve microlib functions and
small helpers before it evolves arbitrary repo patches.

One-sentence mechanism:

FunSearch uses a frozen code LLM as a mutation/crossover operator over one
evolved function, while an evaluator, island population, score-signature
clusters, and best-shot prompts turn stochastic code generation into
executable search.

## Reading Questions

- What is the executable feedback signal?
- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
- What is the population/database/selection mechanism?
- What evidence proves the loop improves over one-shot generation?
- What does the method assume that DynaWorld does not have?

## Mechanism

The user supplies a problem specification:

```text
main(input) -> solution -> evaluate(solution, input) -> score or None
initial evolved function
optional skeleton around that function
optional imports, helper functions, and docstrings
```

The paper's important design move is not "let the LLM write a program." It is
"write the boring program yourself, then let the LLM search the hardest small
piece." In the cap-set example, the fixed skeleton greedily builds a candidate
set. The evolved function is only:

```text
priority(element, n) -> float
```

In the online bin-packing example, the fixed skeleton validates legal bins and
updates remaining capacity. The evolved function is only:

```text
heuristic(item, bins) -> score_per_bin
```

The resulting loop is:

```text
programs_db.get_prompt()
sampler asks LLM for several completions
evaluator executes each candidate function inside the skeleton
if candidate is correct:
    programs_db.register(candidate, scores_per_test)
periodically reset weak islands from strong island elites
```

Important implementation details from the paper and supplement:

- The LLM is frozen. No model training or fine-tuning is required.
- The LLM generates only the function body/signature region being evolved.
- Evaluators reject invalid programs by returning no score.
- Correct candidates are inserted into the program database.
- Prompts are built from prior candidates in the same island.
- The default prompt uses `k = 2` prior functions, ordered from lower scoring
  version to higher scoring version, then asks the model to complete the next
  version. This is "best-shot prompting."
- The population is split into islands. Each island evolves separately.
- Every four hours, the worst half of the islands are wiped and re-seeded with
  elites from surviving islands.
- Within each island, programs are clustered by score signature, meaning the
  tuple of scores over evaluator inputs.
- Cluster sampling favors higher aggregate score; program sampling within a
  cluster favors shorter programs.
- The published hyperparameters include 15 sampler workers, 140 evaluator
  workers, 4 generated samples per prompt, 10 islands, 30 second evaluator
  timeout, 2 GB memory limit, 4 hour island reset period, and 2 prior programs
  per prompt.

This is narrower than AlphaEvolve. AlphaEvolve generalizes from single
functions to larger code regions and richer tasks. FunSearch is therefore the
better implementation starting point for DynaWorld: build the minimum runner
that can evolve one small callable behind one hard evaluator, prove it beats
one-shot Codex, then widen.

## Evaluation

The paper demonstrates FunSearch in two main domains.

### Cap Sets And Admissible Sets

The core math target is to construct large cap sets. A cap set is easy to check
once proposed, but hard to discover at frontier sizes. FunSearch does not
directly output a set in the main formulation. It outputs a priority function
that guides a greedy constructor. That matters because the discovered object is
a compact generative rule rather than a giant list.

Reported evidence:

- FunSearch found a cap set of size 512 in dimension 8, reported as a new state
  of the art for that direct construction task.
- The paper also searches through admissible sets to improve lower bounds on
  cap-set capacity.
- In the statistical analysis, direct n = 8 cap-set discovery was rare: 4 out
  of 140 experiments found the size-512 result.
- Admissible-set routes were more repeatable: all 15 runs across the listed
  dimensions/weights improved the previous best lower bound.
- The discovered functions exposed structure that humans could inspect. The
  authors used one discovered function to notice symmetry, then changed the
  problem skeleton to search directly in a narrower symmetric space.

For `alpha_evolve`, the lesson is that the useful artifact may be a heuristic
that generates good candidates, not the final answer itself. A DynaWorld
microlib might evolve a scheduling policy, candidate-filter function, or
search heuristic whose downstream output is later checked.

### Online Bin Packing

The second domain is online one-dimensional bin packing. The skeleton takes
items in order, filters legal bins, applies an evolved scoring function to the
current item and candidate bins, then chooses the highest-scoring bin.

Reported evidence:

- FunSearch starts from the standard best-fit heuristic.
- It trains/evolves on instances with 5k items and evaluates on 10k-item
  Weibull datasets, so successful functions must generalize across instance
  size.
- Across 10 runs, the discovered heuristics improve over first fit and best
  fit.
- The supplement reports excess bins percentage of 0.44% +/- 0.11% on the
  Weibull 10k dataset, compared with 4.20% for first fit and 3.90% for best
  fit.

For DynaWorld, this is the closest analogue to a microlib like a mixed
same-view/heldout scheduler. The evolved object can be a policy function inside
a fixed training loop, evaluated by finite losses, explicit batch-kind logs,
and leakage checks.

## Ablations And What They Prove

The supplement ablates four components on the symmetric admissible-set task:

- Without the skeleton, the model must generate a more direct ordered list. It
  does worse. Interpretation: do not ask Codex to rediscover boring repo
  scaffolding.
- Without evolution, the system samples repeatedly from only the initial prompt.
  It does worse. Interpretation: a candidate database is not optional if we
  want something stronger than repeated one-shot Codex.
- With less diversity, the system uses a single island. It finds large results
  but misses the full-sized target in all five runs. Interpretation: local
  maxima are real, and one winner thread can prematurely dominate.
- Without merging, prompts use one prior program instead of two. It sometimes
  works, but is weaker overall. Interpretation: prompts should show contrast
  between related candidates, not only the current champion.

The LLM-choice ablation is also important:

- Codey solved more than StarCoder on the reported task.
- StarCoder still produced state-of-the-art improvements.
- Hand-designed random AST mutations were far worse and required manual tuning
  of primitives and corner cases.

For local Codex evolution, the analog is: the exact model matters, but the
runner, skeleton, evaluator, prompt sampling, and archive are first-order
system components. A better model without a database is still a restart loop.

## Why It Beats One-Shot Codex

FunSearch beats one-shot generation because it makes prompt state measurable:

1. Candidate code is executed.
2. Invalid candidates are filtered out.
3. Valid candidates get score vectors over test inputs.
4. Score vectors define clusters.
5. Clusters and islands preserve multiple behavioral modes.
6. Prompt construction gives the model concrete candidate code plus measured
   outcomes.
7. Resets kill weak local searches while keeping strong founders.

The one-shot baseline is not just "one prompt." It is any process that fails to
carry executable candidate history forward. Running `codex exec` 100 times with
the same prompt and no measured archive is closer to the paper's "W/O
Evolution" ablation than to FunSearch.

## DynaWorld Mapping

### Preferred First Shape

The first local runner should be FunSearch-shaped, not AlphaEvolve-wide:

```text
fixed skeleton
one evolved callable or tiny helper file
hard evaluator
program database with score signatures
best-shot prompt with two prior candidates
small island reset policy
```

The current `alpha_evolve/codex_evolver_design.md` assumes patch evolution.
That is still the right long-term shape, but this paper argues for a first
milestone where the "patch" may only replace a small function or isolated helper
module. It will be easier to prove the loop works before letting Codex edit a
trainer or Metal kernel.

### Candidate DynaWorld Skeletons

`mixed_same_view_novel_scheduler` can be made FunSearch-shaped:

```text
fixed loader/trainer smoke
evolved function: choose_next_batch_kind(state, step, metrics) -> kind
evaluator: both kinds executed, separate finite losses, no leakage, no vague
           third manifest format
score signature: same_view_loss, heldout_loss, batch_balance, changed_loc
```

This avoids asking Codex to rewrite the trainer first. The evolved object can
be a scheduler policy or config generator that the existing trainer consumes.

`gaussian_512_promotion_guard` can also be FunSearch-shaped:

```text
fixed promotion smoke
evolved function: should_promote_or_checkpoint(stats, step) -> action
evaluator: catches nonfinite tensors, checkpoints pre-promotion, reaches 512px
           smoke without corrupting optimizer state
score signature: finite, promotion_step, nan_source_found, loss_delta
```

`star_uvt_feature_rgb_handoff` is harder. The interesting target spans kernel
and gradient code, so it is less naturally a one-function FunSearch problem.
A safer FunSearch-shaped subproblem would be:

```text
fixed benchmark runner
evolved helper: select_or_partition_gradient_path(feature_shape, colorizer_spec)
evaluator: F4/F32 parity, nonzero feature/colorizer grads, zero overflow,
           lower backward ms
```

This will not solve the final trainable STAR feature path alone, but it gives
the runner a tractable first code-search surface.

### Program Database Schema Update

FunSearch suggests that `programs.jsonl` should store both scalar metrics and a
score signature:

```json
{
  "candidate_id": "cand_000042",
  "island": "mixed_scheduler_03",
  "parent_ids": ["cand_000017", "cand_000031"],
  "status": "accepted",
  "score": 0.73,
  "signature": {
    "same_view_recon_bucket": "finite_low",
    "heldout_view_recon_bucket": "finite_mid",
    "leakage": "none",
    "batch_balance": "both"
  },
  "metrics": {
    "same_view_recon": 0.184,
    "heldout_view_recon": 0.291,
    "changed_loc": 38
  },
  "function_text_path": "candidates/cand_000042/evolved.py",
  "patch_path": "candidates/cand_000042/patch.diff"
}
```

The important part is not the exact buckets. It is making behavioral diversity
explicit enough that the sampler can preserve non-identical candidates.

### Prompt Sampling Update

A FunSearch-like Codex prompt should not include a single winner. It should
include two candidates from the same island:

```text
Candidate v0:
    lower score, simpler or different behavior
Candidate v1:
    higher score
Task:
    produce v2 that improves the hard evaluator while preserving invariants
```

For DynaWorld, the prompt should include score vectors and failure reasons:

```text
v0 metrics:
    same_view_recon finite, heldout missing
v1 metrics:
    both finite, leakage tripwire too weak
Known rejects:
    changing manifest format, collapsing log keys, skipping feature grads
```

This is more concrete than "learn from prior attempts" and should reduce
prompt bloat compared with dumping full loose notes.

## Failure Modes

### Skeleton Bias

The skeleton focuses search, but it also prevents discoveries outside its
shape. FunSearch found symmetry partly because the priority-function skeleton
made the structure visible, but a bad skeleton can hide the real move.

DynaWorld implication:

- Start with skeletonized microlibs for proof of runner.
- Keep an explicit `skeleton_bias` note per problem.
- If islands plateau, do not only increase samples. Consider a different
  evolved interface.

### Evaluator Overfit

FunSearch's domains have clean evaluators: cap-set validity and bin-packing
legality are crisp. DynaWorld evaluators are often proxies: 1-step smoke,
20-step overfit, synthetic timing, saved JSON rows. A candidate can hack them
by changing frame count, reducing resolution, disabling gradients, weakening
logs, or bypassing media generation.

Hard local requirements:

- forbidden path/pattern checks
- invariant checks before scores
- separate "correctness" and "quality" metrics
- evaluator file immutability unless the microlib is explicitly evaluator
  generation
- holdout variants that are not shown in prompts

### Stochastic Rarity

The cap-set result was rare in direct form: 4 out of 140 experiments. This
warns against interpreting early failures as a full negative. But it also warns
against claiming success from one lucky seed.

DynaWorld implication:

- For cheap microlibs, use repeated seeds in the evaluator.
- For expensive microlibs, require at least a replay of the winning patch from
  a clean worktree before promotion.
- Store failed candidates because they characterize the search surface.

### One Function Is Not Always Enough

FunSearch's strength comes from narrow function evolution. DynaWorld's hardest
problems may require coordinated changes across configs, trainers, renderers,
and tests. Starting narrow is good, but the runner must eventually support
patch-level AlphaEvolve-style tasks.

The local sequence should be:

1. function-level microlib proof
2. helper-file microlib proof
3. patch-level microlib with strict allowed paths
4. only then broader trainer/kernel work

## Falsification Tests

### Test 1 - One-Shot Codex Versus FunSearch-Shaped Archive

Pick the mixed scheduler policy as a toy target.

Procedure:

1. Define a tiny fixed skeleton with an evolved scheduling function.
2. Run `codex exec` 20 times with the same initial prompt and no archive.
3. Run 20 candidates with a tiny programs database and best-shot prompts using
   two prior candidates.
4. Keep evaluator, model, allowed paths, and budget equal.

Support for FunSearch mapping:

- archive run produces more accepted candidates or better score signatures
- archive run avoids repeated known reject patterns
- archive run finds both-batch finite behavior faster

Falsification:

- archive run performs no better than one-shot and mostly adds overhead
- prompt history causes bloat, confusion, or repeated regression

### Test 2 - Skeleton Interface Sweep

For one microlib, define two evolved interfaces:

```text
narrow: choose_next_batch_kind(state) -> kind
wider: build_batch_plan(state) -> list[batch_request]
```

Run the same candidate budget.

Support:

- narrow interface gets correctness quickly but plateaus on quality
- wider interface has more invalid candidates but discovers better tradeoffs

Use:

- choose the narrow interface for proof of runner
- promote to the wider interface only if plateau evidence is clear

### Test 3 - Signature Clustering Beats Scalar Ranking

Run selection two ways:

```text
A: sort by one scalar score
B: cluster by signature, sample high-score clusters, favor shorter code inside
```

For DynaWorld, useful signatures might include:

- both batch kinds executed
- leakage status
- finite status
- loss bucket
- changed LOC bucket
- backward timing bucket

Support:

- clustered selection preserves more valid behavioral variants
- scalar selection collapses to one hacky family

Falsification:

- clusters add complexity but do not improve accepted-candidate diversity or
  final scores.

## Design Decisions For `alpha_evolve`

1. Add a function-level mode before full patch evolution.
2. Treat a microlib as a skeleton plus evolved surface, not just a prompt.
3. Store score signatures in `programs.jsonl`.
4. Include two prior candidates in prompts by default.
5. Add islands only after a serial archive loop works; use a small number first,
   such as 3, not the paper's 10.
6. Use island reset by candidate count or wall time. Four hours is paper-scale;
   local proof runs should probably reset every 10-25 accepted candidates.
7. Favor shorter accepted candidates inside equivalent score clusters.
8. Keep exact evaluator inputs hidden or varied where metric hacking is easy.
9. Do not let a failed expensive microlib disprove the runner. Prove the runner
   first on a cheap skeletonized target.

## Notes For Future Papers

- CodeEvolve should be read as the bridge from FunSearch's one-function
  setting to AlphaEvolve-style repo patches.
- Evolution of Heuristics and LLaMEA should clarify whether the island and
  archive mechanics used here are standard enough to copy directly.
- Eureka and Voyager should be read for how they store skills/rewards and
  failure feedback, but FunSearch is stricter because every improvement is an
  executable program with an external evaluator.
- Agentless should later challenge whether the archive/island machinery beats a
  simpler locate-then-patch baseline for DynaWorld code tasks.
