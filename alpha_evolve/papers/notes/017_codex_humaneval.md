# 017 - Evaluating Large Language Models Trained on Code

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2107.03374
    https://arxiv.org/pdf/2107.03374
    https://github.com/openai/human-eval

Implementation artifacts inspected:
    https://github.com/openai/human-eval/blob/master/README.md
    https://github.com/openai/human-eval/blob/master/human_eval/evaluation.py
    https://github.com/openai/human-eval/blob/master/human_eval/execution.py

Bibliographic metadata:
    Authors: Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique
    Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda,
    Nicholas Joseph, Greg Brockman, Alex Ray, Raul Puri, Gretchen Krueger,
    Michael Petrov, Heidy Khlaaf, Girish Sastry, Pamela Mishkin, Brooke Chan,
    Scott Gray, Nick Ryder, Mikhail Pavlov, Alethea Power, Lukasz Kaiser,
    Mohammad Bavarian, Clemens Winter, Philippe Tillet, Felipe Petroski Such,
    Dave Cummings, Matthias Plappert, Fotios Chantzis, Elizabeth Barnes,
    Ariel Herbert-Voss, William Hebgen Guss, Alex Nichol, Alex Paino, Nikolas
    Tezak, Jie Tang, Igor Babuschkin, Suchir Balaji, Shantanu Jain, William
    Saunders, Christopher Hesse, Andrew N. Carr, Jan Leike, Josh Achiam,
    Vedant Misra, Evan Morikawa, Alec Radford, Matthew Knight, Miles Brundage,
    Mira Murati, Katie Mayer, Peter Welinder, Bob McGrew, Dario Amodei,
    Sam McCandlish, Ilya Sutskever, Wojciech Zaremba.
    First arXiv submission: 2021-07-07.
    Version inspected: arXiv v2, 2021-07-14.

Why this paper matters for alpha_evolve:
    This is the functional-correctness and sampling metric foundation for the
    rest of the code-agent papers. It gives us:

```text
HumanEval:
    hand-written code tasks
    unit-test based correctness

pass@k:
    measure whether any of k samples solves the task

sample selection:
    distinguish oracle best-of-k from realistic rank-one selection

sandboxing:
    generated code is untrusted and must be isolated
```

    For `alpha_evolve`, this is the bridge from "ask Codex once" to "sample a
    population of candidate patches and score them correctly."

One-sentence mechanism:
    Fine-tune GPT models on GitHub code to create Codex, evaluate generated
    Python function completions with unit tests on HumanEval, and report
    functional correctness using an unbiased pass@k estimator over multiple
    generated samples.

## Reading Questions

- What is the executable feedback signal?
  Unit tests. A generated completion is correct if it passes the tests for a
  HumanEval task. The official harness executes generated code and reports
  `passed`, `timed out`, or `failed`.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Code samples. The paper samples multiple completions per prompt. There is no
  interactive agent trajectory and no long-term memory.

- What is the population/database/selection mechanism?
  The population is the set of sampled completions per problem. Selection can be
  oracle unit-test selection, mean log-probability ranking, random choice, or
  other heuristics. The paper carefully separates best-of-k from practical
  single-sample selection when tests are not available.

- What evidence matters?
  Codex-12B solves 28.81 percent of HumanEval with pass@1. Codex-S solves 37.7
  percent with one sample in the headline result. Repeated sampling is powerful:
  the abstract reports 70.2 percent solved with 100 samples, and the paper's
  Figure 1 reports 77.5 percent for Codex-S when selecting a 100-sample solution
  that passes tests. Mean log-probability selection from 100 samples reaches
  44.5 percent in that figure.

- What does this assume that DynaWorld does not yet have?
  Small, deterministic, standalone functions with fast unit tests. DynaWorld
  tasks are repo-level and may include training, rendering, config behavior, or
  docs. We need to carve out HumanEval-like microtasks where functional
  correctness is cheap before applying pass@k-style sampling.

## Functional Correctness

The paper rejects match-based code metrics as the main judge. Exact match and
BLEU-style scores cannot capture the many programs that are functionally
equivalent to a reference solution.

The core evaluation rule:

```text
completion is correct if it passes the task's unit tests
```

This is the right local principle too. Do not rank evolved code by:

```text
diff similarity to a reference
looks concise
model preference
note quality
prompt compliance alone
```

Rank by executable behavior first. Use style/model judgments only as secondary
signals after behavior gates.

The paper also shows BLEU overlap is not separable between correct and wrong
solutions. For `alpha_evolve`, this generalizes to all text-only patch scoring:

```text
model says patch is elegant:
    weak signal

patch passes visible and hidden tests:
    strong signal
```

## HumanEval

HumanEval consists of:

```text
164 hand-written Python programming problems
function signature
docstring
function body prompt
unit tests
average 7.7 tests per problem
```

The tasks assess:

```text
language comprehension
reasoning
algorithms
simple mathematics
```

They are hand-written because the Codex training data includes a large fraction
of public GitHub. Existing programming-contest tasks may have public solutions
in the training data. Hand-writing does not guarantee no contamination, but it
reduces direct copying risk.

Local transfer:

```text
alpha_evolve/tasks/humaneval_like/
    should be hand-authored local microtasks
    should avoid directly copying public benchmark tasks
    should include hidden tests
    should produce fast pass/fail signals
```

This is useful before repo-scale DynaWorld targets. A simple microlib task might
ask for a scheduler helper, score parser, context-window builder, or patch
normalizer with unit tests.

## pass@k

The paper defines pass@k as the probability that at least one of k generated
samples passes tests for a problem.

Because directly sampling k and counting solved problems can be high variance,
the paper generates n samples, counts c correct samples, and uses the unbiased
estimator:

```text
pass@k = 1 - C(n - c, k) / C(n, k)
```

where:

```text
n:
    total samples for the problem

c:
    number of samples that pass tests

k:
    number of samples considered
```

The stable implementation in the paper and official harness avoids huge
combinations by computing:

```text
if n - c < k:
    return 1.0

return 1.0 - product(1.0 - k / arange(n - c + 1, n + 1))
```

Important warning:
    Do not estimate pass@k as `1 - (1 - pass@1)^k` from empirical pass@1. The
    paper shows this estimator is biased.

Local implication:
    If `alpha_evolve` samples many patches per task, report:

```text
pass@1:
    selected patch success

pass@k_oracle:
    whether any of k sampled patches passed hidden evaluator

ranker_gap:
    oracle pass@k minus selected patch success
```

The ranker gap is crucial. If any sampled patch works but the ranker picks the
wrong one, generation is not the bottleneck.

## Sampling Temperature

The paper finds that optimal sampling temperature depends on k:

```text
small k / pass@1:
    lower temperature is better

large k / pass@100:
    higher temperature is better because diversity matters
```

For a 679M Codex model:

```text
pass@1 optimal temperature:
    T = 0.2

pass@100 optimal temperature:
    T = 0.8
```

Local transfer:

```text
one-shot Codex baseline:
    use conservative prompt/settings

patch population search:
    intentionally increase diversity
    deduplicate patches before evaluation
    separate generation temperature from final ranker
```

This also means we should not judge a prompt by pass@1 only if the intended
evolver will sample many candidates.

## Results

HumanEval headline results:

```text
GPT-3 family:
    near 0 percent

GPT-J 6B:
    11.4 percent in abstract discussion
    table reports 11.62 pass@1 and 27.74 pass@100

Codex-300M:
    13.17 pass@1
    36.27 pass@100

Codex-12B:
    28.81 pass@1
    46.81 pass@10
    72.31 pass@100

Codex-S:
    37.7 percent pass@1 in Figure 1 / headline discussion
```

Sampling and selection:

```text
100 samples with unit-test oracle:
    abstract reports 70.2 percent solved
    Figure 1 reports Codex-S 77.5 percent solved

100 samples with mean log-probability selection:
    44.5 percent solved in Figure 1
```

The takeaway for local evolution is that sampling can create enough solution
diversity to make selection the bottleneck. This connects directly to Agentless,
which found that its sample pool contained more potential fixes than its final
selector recovered.

## Selection Without Oracle Tests

The paper studies the setting where many samples can be generated but only one
sample can be returned. Mean token log-probability outperforms random selection,
while sum log-probability can be worse than random because it favors short
outputs differently.

Local implication:
    For patch selection, do not assume the highest-probability patch is best.
    Use it as one feature in the ranker:

```text
features:
    visible evaluator score
    hidden-like generated tests
    syntax/import guard
    patch size
    changed-file scope
    duplicate/majority cluster size
    model confidence/logprob if available
    failure-risk labels
```

The paper gives us the metric language to evaluate rankers:

```text
best-in-sample:
    hidden oracle over all sampled patches

selected:
    patch chosen by cheap ranker

ranker_gap:
    best-in-sample score - selected score
```

## Sandbox

The paper emphasizes that executing generated code is dangerous. The authors
used a sandbox designed to prevent generated programs from modifying,
persisting on, accessing sensitive resources on, or exfiltrating data from a
host/network. They used gVisor and network firewalling in their infrastructure.

The official HumanEval README preserves this warning. The execution call is
deliberately commented out so users must acknowledge that generated code is
untrusted.

Local transfer:
    Every `alpha_evolve` candidate must run in an isolated runtime:

```text
minimum:
    temp directory or git worktree
    no unrelated dirty files
    timeout
    artifact cleanup

better:
    container or sandbox
    network disabled for candidate code
    resource limits
    explicit allowlist for evaluator commands
```

DynaWorld adds a special problem: some tests need local GPU/Metal state or
large artifacts. The runtime adapter must record what isolation level was used
so scores remain comparable.

## Official HumanEval Harness

The inspected `human_eval/evaluation.py` does three important things:

```text
reads JSONL samples:
    task_id
    completion

executes correctness checks:
    in workers
    with per-completion timeout

writes per-sample results:
    result
    passed
```

It computes pass@k only when every problem has at least k samples. This matters:
pass@100 with missing samples for some tasks is not comparable.

Local transfer:
    The `alpha_evolve` benchmark runner should refuse to report pass@k unless
    every task has at least k completed candidate evaluations, or it should
    explicitly report the eligible subset.

Suggested local result schema:

```text
{
  "task_id": "...",
  "candidate_id": "...",
  "completion_id": 17,
  "patch_path": "...",
  "result": "passed|failed|timed_out|guard_failed",
  "passed": true,
  "visible_score": 1.0,
  "hidden_score": 1.0,
  "duration_sec": 2.31
}
```

## What Transfers To `codex exec`

The first local sampling loop can be exactly HumanEval-shaped:

```text
for task in tasks:
    for sample_id in 1..n:
        codex exec "<task prompt + strict patch schema>"
        parse patch
        run guard/evaluator
        write result JSONL

compute:
    pass@1 selected
    pass@k oracle
    ranker_gap
    cost per solved task
```

Do not collapse samples before writing results. Deduplication can happen as a
secondary analysis, but the raw sample count and correctness count are needed
for honest pass@k.

## Microlibs Suggested By This Paper

```text
functional_correctness/
    Standard task result schema and pass/fail semantics.

pass_at_k/
    Unbiased pass@k estimator plus validation for missing samples.

sample_runner/
    Runs N Codex samples per task and writes JSONL.

sample_deduper/
    Normalizes patches for cluster/majority analysis without corrupting raw
    pass@k accounting.

oracle_selector/
    Offline analysis of best-in-sample using hidden evaluator.

ranker_gap/
    Compares selected candidate to oracle best-in-sample.

sandbox_policy/
    Runtime isolation, timeout, network, and resource-limit metadata.

humaneval_like_tasks/
    Small local function/microlib tasks with fast unit tests.
```

## Target Problems In This Repo

Good HumanEval-like DynaWorld tasks:

```text
score parser:
    parse evaluator logs into structured metrics

context window builder:
    turn file and line spans into bounded snippets

patch normalizer:
    normalize diffs for majority clustering

failure classifier helper:
    map evaluator outcomes to labels

config resolver helper:
    normalize a small config section with explicit edge cases

anti-pattern detector:
    detect P1/P2/P3 patterns in small code snippets
```

These are better first pass@k tasks than:

```text
full trainer refactor
renderer kernel optimization
GPU training objective
multi-file architecture rewrite
```

Once pass@k and ranker-gap machinery works on small tasks, move to SWE-style
repo tasks.

## Red-Team Notes

Risk: Unit tests are incomplete.
    Functional correctness is only as strong as the tests. Hidden tests and
    property tests are needed for local promotion.

Risk: pass@k can hide selection weakness.
    High oracle pass@k does not mean the system can choose a good patch without
    hidden tests. Always report selected pass@1 and ranker gap.

Risk: sample budget comparisons can be unfair.
    pass@k requires comparable n and k. Missing samples or different timeouts
    can distort results.

Risk: generated code is untrusted.
    Sandbox every candidate. Generated tests are code too.

Risk: HumanEval is too small and standalone.
    It is useful for metric machinery, not enough to prove repo-level repair.

Risk: contamination.
    Public benchmark tasks can enter model training data. Local hand-authored
    tasks reduce but do not eliminate leakage concerns.

## Local Falsification Tests

1. pass@k estimator test:

```text
construct synthetic result arrays with known n, c, k
compare implementation to official formula
reject biased 1 - (1 - p)^k shortcut
```

2. missing sample test:

```text
task A has 10 samples
task B has 9 samples
request pass@10
runner must refuse or report eligible subset explicitly
```

3. ranker gap smoke:

```text
sample 10 patches for a toy task
make patch 7 the only hidden-pass patch
cheap ranker chooses patch 2
report oracle pass@10 = true and selected pass@1 = false
```

4. sandbox failure:

```text
candidate tries network/file escape/long sleep
runner times out or blocks
archive result as sandbox violation
```

5. BLEU/text score trap:

```text
two patches:
    one similar to reference but wrong
    one different but correct
rank by tests, not text similarity
```

## Design Consequences

After this paper, the implementation stack needs a metric layer before a search
layer:

```text
sample_runner
result_jsonl
functional_correctness
pass_at_k
oracle_selector
ranker_gap
sandbox_policy
```

AlphaEvolve-style evolution without honest pass@k and ranker-gap reporting will
confuse generation ability, evaluator quality, and selection quality. The first
`codex exec` experiments should therefore run many samples on tiny local tasks,
not one sample on a giant trainer change.

## Open Questions For Later Papers

- Does the program-synthesis scaling paper add better prompt/task structure for
  natural-language-to-code tasks?
- Does AlphaCode show how to filter massive sample pools when running hidden
  tests on every sample is too expensive?
- Does CodeT provide a practical generated-test agreement signal that improves
  ranker gap?
- What is the smallest local task suite where pass@k predicts useful repo-level
  repair quality?
