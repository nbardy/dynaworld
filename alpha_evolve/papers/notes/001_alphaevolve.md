# 001 - AlphaEvolve: A coding agent for scientific and algorithmic discovery

Status:
    first-pass

Primary sources:

- arXiv page: https://arxiv.org/abs/2506.13131
- PDF: https://storage.googleapis.com/deepmind-media/DeepMind.com/Blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/AlphaEvolve.pdf
- DeepMind launch post: https://deepmind.google/blog/alphaevolve-a-gemini-powered-coding-agent-for-designing-advanced-algorithms/

Why this paper matters for `alpha_evolve`:

AlphaEvolve is the direct model for the local idea: an LLM proposes code
changes; executable evaluators score candidates; a program database stores
solutions and feedback; prompts are sampled from that database; evolution
continues until useful programs emerge. The translation to DynaWorld should
preserve the evaluator-first discipline and avoid turning this into a vague
multi-agent brainstorming loop.

One-sentence mechanism:

AlphaEvolve uses state-of-the-art LLMs as mutation operators over code, grounds
the search with automatic scalar evaluators, and uses a diversity-preserving
program database to choose parent and inspiration programs for future prompts.

## Reading Questions

- What makes AlphaEvolve stronger than plain one-shot code generation?
- What exactly is the evaluator contract?
- Why does the paper emphasize code, not natural-language hypotheses?
- How should DynaWorld choose the unit of evolution: function, file, patch, or
  microlib?
- What prevents metric hacking?
- What must be true before expensive multi-hour evaluators make sense?

## Mechanism

AlphaEvolve starts from a user-provided task:

```text
initial program + evolve-marked components + evaluate(solution) -> metrics
```

The loop is:

```text
parent_program, inspirations = database.sample()
prompt = prompt_sampler.build(parent_program, inspirations)
diff = llm.generate(prompt)
child_program = apply_diff(parent_program, diff)
metrics = evaluator.execute(child_program)
database.add(child_program, metrics)
```

Important details:

- The evaluator returns a dictionary of scalar metrics, conventionally
  maximized.
- Evolved code can be tiny or large. The paper contrasts FunSearch's
  single-function, 10-20-line regime with AlphaEvolve's ability to evolve
  whole files or hundreds of lines.
- Code regions can be marked with EVOLVE-BLOCK comments, but the broader idea
  is a controlled edit surface plus a complete runnable skeleton.
- Prompts include prior programs, current program, scores, rendered evaluation
  results, fixed context, and sometimes literature or equations.
- Output can be SEARCH/REPLACE diffs or whole code blocks.
- The system uses multiple LLMs with a throughput/quality tradeoff: a fast model
  for many candidates and a stronger model for occasional larger jumps.
- Evaluation is staged: cheap early filters, then larger tests only for
  promising candidates.
- Scores are multi-objective. Even when there is one main metric, auxiliary
  metrics can preserve structurally different high-performing programs.
- The program database is inspired by MAP-Elites and island population models.
- The system is asynchronous and throughput-oriented, not optimized for one
  candidate's latency.

## Evaluation

The paper's strongest claim is not that the LLM understands the problem in a
human sense. The claim is that executable evaluation turns code proposals into
scientific search.

Reported domains:

- Matrix multiplication algorithms. The paper reports improvements for 14
  matrix multiplication targets and a rank-48 algorithm for 4x4 complex-valued
  matrix multiplication.
- Open mathematical construction problems. The paper reports matching best
  known constructions on about 75 percent of a 50-plus-problem set and improving
  about 20 percent.
- Google infrastructure. Reported applications include data-center scheduling,
  TPU arithmetic circuit simplification, matrix multiplication kernels, and
  attention runtime.

Evaluation observations:

- The matrix multiplication case uses a search algorithm as the evolved object,
  not just a direct candidate tensor. This matters for DynaWorld: a microlib may
  evolve a search heuristic that finds kernels/configs, not only the final
  kernel.
- Some problems benefit from human seeding. The loop is not purely autonomous;
  a good initial representation and evaluator can change the search space.
- Evaluation can be expensive. The paper explicitly allows hours and large
  parallel budgets, but only because candidates are automatically judged.

## Why It Beats One-Shot Codex

The advantage is not just "more samples." It is structured reuse of feedback:

1. Each candidate is run.
2. Scores and outputs are stored.
3. Future prompts see prior programs and their measured behavior.
4. The database keeps diverse elites alive.
5. Evaluation cascades prevent obviously bad variants from consuming the full
   budget.

For local `codex exec`, this means a useful runner needs a database before it
needs many agents. Without the database, repeated Codex calls are just an
expensive restart loop.

## DynaWorld Mapping

### Unit Of Evolution

Do not start by evolving arbitrary trainer code. Start with microlibs:

```text
problem contract + allowed paths + evaluator cascade + score schema
```

Good candidates:

- STAR UVT feature RGB-gradient handoff.
- Mixed same-view plus heldout scheduler.
- Gaussian 512px promotion guard.
- V-JEPA/F32 multicam benchmark validators.

Bad first candidates:

- final world-token architecture
- long cloud training runs
- visual-only media judgments
- broad multi-agent role prompts

### Evaluator Shape

Every microlib needs:

```text
evaluate(candidate_worktree) -> {
  "correct": bool,
  "finite": bool,
  "primary_score": float,
  ...
}
```

For costs where smaller is better, convert to maximization:

```text
neg_backward_ms = -backward_ms
neg_changed_loc = -changed_loc
```

The evaluator should include hard gates before soft scores:

- scope ok
- no forbidden edits
- no changed evaluator unless the microlib is an evaluator microlib
- parity/test pass
- no nonfinite values

### Program Database

Use `outputs/alpha_evolve/<problem>/<run_id>/programs.jsonl`.

Store both winners and informative failures. Future prompts should include:

- one global elite
- one island elite
- one near miss
- one failure with a clear lesson

Do not include the entire history in every prompt.

### Prompt Sampling

The paper's rich prompt lesson maps directly:

- include current metrics, not vibes
- include failure snippets, not just winners
- include fixed problem context and allowed paths
- include evaluator command snippets
- include non-goals in each prompt

For DynaWorld, prompt bloat is a real risk because the repo already has dense
notes. The prompt sampler should select sharp evidence, not dump all docs.

### Search Space

The paper's "flexibility in choosing abstraction" is important. In DynaWorld,
there are multiple valid evolved objects:

- a direct patch to a kernel/trainer helper
- a search script that discovers configs
- a guard/diagnostic that makes a later search safe
- a generated evaluator or manifest validator

For STAR UVT feature work, evolving the final Metal path may be too brittle at
first. A safer first pass is to evolve experiment-side prototypes and promote
only after parity/timing gates.

## Failure Modes

### Metric Hacking

If a candidate can reduce runtime by dropping gradients, reducing frames,
lowering resolution, changing the target, or skipping media, it will eventually
do that. The evaluator must check invariants.

For STAR feature:

- feature gradients must be present
- colorizer gradients must be present for RGB handoff
- frame count, tube count, feature dim, target size, and loss kind must match
- overflow fallback must be reported

### Evaluator Leakage

Candidate patches must not edit evaluator scripts unless the microlib is
specifically about evaluator design. Keep evaluator files outside allowed paths
for implementation candidates.

### Expensive False Positives

Two-step smokes can select variants that fail at 20 steps. A candidate should
advance through stages:

```text
static -> parity -> tiny timing -> 20-step smoke -> promoted run
```

### Prompt Collapse

If the database only feeds the current best program, candidates will converge
to local tweaks. Maintain islands or behavior bins.

Possible STAR bins:

- RGB-gradient handoff
- feature-gradient reduction
- fixed-bin backward
- support pruning
- memory valve

### Dirty Tree Contamination

DynaWorld's active tree is often dirty. AlphaEvolve assumes a controlled
program version. Local implementation must use disposable worktrees or clean
snapshots, not the live user tree.

## Falsification Tests For Local Runner

### Test 1: Does Evolution Beat One-Shot Codex?

Setup:

- Choose STAR UVT feature RGB handoff.
- Run 1 one-shot Codex candidate with the microlib prompt.
- Run 10 evolved candidates with database feedback.
- Same allowed paths and Stage 0-1 evaluator.

Support:

- evolved loop finds more Stage 1 passing candidates or lower backward proxy.

Weakens:

- evolved loop does not outperform one-shot and mostly repeats failures.

Implication:

- If weak, improve prompt/database first, not model count.

### Test 2: Does Island Diversity Matter?

Setup:

- Run greedy-parent evolution versus island-sampled evolution for the same
  candidate budget.

Support:

- islands produce at least two structurally different passing candidates.

Weakens:

- all islands collapse to the same patch family.

Implication:

- Need better behavior descriptors or stronger prompt diversity.

### Test 3: Can The Evaluator Detect Metric Hacking?

Setup:

- Hand-write bad candidates that skip feature gradients, lower frame count, or
  change loss target.

Support:

- Stage 0/1 rejects all of them.

Weakens:

- any bad candidate scores as an improvement.

Implication:

- Do not run evolution until the evaluator is fixed.

## Design Decisions For `alpha_evolve/`

1. First implementation should be single-microlib.
2. Candidate generation must run in a disposable worktree.
3. Candidate DB is required before parallelism.
4. The runner should save Codex events, final message, patch, changed files,
   evaluator JSONs, and status.
5. Scoring should be multi-objective even if selection has one primary metric.
6. Use `codex exec`, not current `codex -p`, because `-p` is profile in this
   CLI install.

## Open Questions

- Should DynaWorld use EVOLVE-BLOCK comments in code, or keep allowed-path
  patch scopes only?
- How much repo context should the prompt sampler include before it hurts?
- Should failures be summarized by Codex after evaluation, or by deterministic
  evaluator labels?
- What is the smallest STAR feature evaluator that is fast enough for 20+
  candidates but predictive of real 20-step training?
- Should runner metadata live in tracked `alpha_evolve/` or ignored
  `outputs/alpha_evolve/` only?

## Next Paper Links

FunSearch is the next read because AlphaEvolve frames itself as a substantial
extension of FunSearch. CodeEvolve and EoH should come after that because they
are closer to an implementable open loop.
