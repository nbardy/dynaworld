# 010 - Self-Refine: Iterative Refinement with Self-Feedback

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2303.17651
    https://arxiv.org/pdf/2303.17651
    https://github.com/madaan/self-refine
    https://selfrefine.info/

Implementation artifacts inspected:
    https://github.com/madaan/self-refine/blob/main/README.md
    https://github.com/madaan/self-refine/blob/main/src/pie/run.py
    https://github.com/madaan/self-refine/blob/main/src/pie/feedback.py
    https://github.com/madaan/self-refine/blob/main/src/pie/task_iterate.py
    https://github.com/madaan/self-refine/blob/main/src/readability/readability.py

Bibliographic metadata:
    Authors: Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan,
    Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye,
    Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine
    Hermann, Sean Welleck, Amir Yazdanbakhsh, Peter Clark.
    First arXiv submission: 2023-03-30.
    Latest arXiv version inspected: v2, 2023-05-25.
    Venue/context: NeurIPS 2023.

Why this paper matters for alpha_evolve:
    Self-Refine is the minimal feedback-refine pattern:

```text
generate -> feedback -> refine -> repeat
```

    It matters because an `alpha_evolve` runner will need this pattern inside
    candidate repair, prompt improvement, generated-test improvement, loss-shaper
    improvement, and documentation/prompt cleanup. It is simpler than Reflexion:
    no environment trajectory, no actor/evaluator/self-reflection split, and no
    explicit long-term memory. The same model generates the output, critiques it,
    and revises it.

    That simplicity is both useful and dangerous. For repo code, same-model
    feedback is allowed to guide a retry, but it should not replace execution
    feedback, hidden validation, or baseline comparison.

One-sentence mechanism:
    Use one LLM with three prompts, initial generation, feedback, and refinement,
    then append prior drafts plus feedback into the next refinement prompt until
    a task-specific stop condition or iteration cap fires.

## Reading Questions

- What is the executable feedback signal?
  Mostly none. The core method uses model-generated natural-language feedback,
  sometimes with model-generated scalar scores or stop indicators. Some tasks
  are later evaluated by humans, GPT-4 preference, automatic task metrics, or
  code optimization metrics, but those metrics are not generally the feedback
  source inside the loop.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Outputs. Depending on the task, the output is text, math reasoning, code,
  optimized code, readable code, acronyms, or constrained sentences. There is no
  population. The loop rewrites one candidate repeatedly.

- What is the population/database/selection mechanism?
  None in the main method. It keeps the history of drafts and feedback in the
  prompt, and returns the last refinement. For multi-aspect feedback tasks, the
  paper notes that quality is not always monotonic, so numerical aspect scores
  can help select a balanced output. That is the closest thing to selection.

- What evidence proves the loop improves over one-shot generation?
  The paper reports improvements across seven tasks using GPT-3.5, ChatGPT, and
  GPT-4. Main table examples include GPT-4 code optimization from 27.3 to 36.0,
  ChatGPT code readability from 27.7 to 63.1, and GPT-4 constrained generation
  from 15.0 to 45.0. The paper also compares refinement to k independent samples
  and reports that Self-Refine outputs are preferred over all k initial outputs
  in a harder 1-vs-k setting.

- What does the method assume that DynaWorld does not have?
  It assumes the model can accurately critique its own output. That is not safe
  for DynaWorld code changes, where the hard questions are often invisible to
  the model until a command runs: tensor shapes, data contracts, renderer
  dispatch, hidden baselines, or runtime behavior.

## Mechanism

Self-Refine has three prompts:

```text
p_gen:
    generate initial output y_0 from input x

p_fb:
    generate feedback fb_t from input x and current output y_t

p_refine:
    generate refined output y_{t+1} from x, current output, prior outputs,
    and feedback
```

The paper describes the loop as:

```text
y_0 = M(p_gen || x)

for t = 0, 1, ...
    fb_t = M(p_fb || x || y_t)
    if stop(fb_t, t):
        break
    y_{t+1} = M(p_refine || x || y_0 || fb_0 || ... || y_t || fb_t)

return y_t
```

The feedback prompt is the important piece. The paper stresses that feedback
must be:

```text
actionable:
    contains a concrete change likely to improve the output

specific:
    points to concrete parts of the output to change
```

This maps cleanly to local Codex prompts. A vague critique like "make it faster"
is not enough. A useful critique says:

```text
The slow path is the repeated all-frame renderer invocation in function X.
Keep the API stable, but cache Y before loop Z.
```

For code evolution, the local refinement prompt should separate:

```text
model critique:
    hypothesis about what to change

evaluator feedback:
    observed stdout/stderr/metric facts

repo constraints:
    files allowed, invariants, smoke gates, hidden gates not shown
```

Self-Refine by itself only supplies the first part.

## Difference From Reflexion

Reflexion:

```text
environment/evaluator gives feedback
reflection compresses failed trial into memory
next trial uses memory
```

Self-Refine:

```text
same model critiques current output
same model revises current output
history stays inside the prompt
```

Practical distinction:

```text
Reflexion is better for failed attempts with external feedback.
Self-Refine is better for improving an output along soft or multi-aspect goals.
```

For `alpha_evolve`, Self-Refine should not be the main evaluator loop. It should
be a helper used when the task is:

```text
make prompt clearer
make generated feedback more actionable
make generated test suite broader
make a candidate patch smaller
make notes or JSON schema cleaner
improve code readability after a hard gate already passes
```

It is weaker for:

```text
prove code correctness
verify benchmark improvement
diagnose numerical regressions
detect hidden data leakage
escape local minima that need a new algorithm family
```

## Implementation Artifacts

The official repo is organized by task, with each task exposing the same shape:

```text
Init:
    initial output prompt

Feedback:
    critique prompt

Iterate:
    refinement prompt
```

The README explicitly describes the three prompt types and points to task
directories such as:

```text
src/acronym
src/commongen
src/gsm
src/pie
src/readability
src/responsegen
src/sentiment_reversal
```

### PIE Code Optimization

The `src/pie/run.py` implementation is the most relevant to DynaWorld.

Loop sketch:

```text
while attempts < max_attempts:
    if first attempt:
        fast_code = PieInit(slow_code)
    else:
        fast_code = PieIterate(slow_code, feedback)

    feedback = PieFeedback(fast_code)
    log fast_code, feedback, slow_code, attempt

    if feedback says code is not slow:
        break

    slow_code = fast_code
```

The feedback module asks:

```text
# Why is this code slow?
```

The iterate module asks:

```text
# Improved version:
```

and includes the current code plus feedback. The ablations can switch feedback
to:

```text
naive:
    "It could be faster"

none:
    ""
```

That is exactly the local ablation to run later:

```text
codex exec with specific failure feedback
codex exec with generic failure feedback
codex exec with no feedback
```

The key limitation is visible in the code: the loop asks the model whether code
is still slow. It does not run a profiler inside the refinement loop. The paper
evaluates optimization later, but the inner loop's feedback is model-side.

Local rule:

```text
Self-Refine feedback can propose a performance fix.
The evaluator must run the benchmark/profiler before ranking or promotion.
```

### Code Readability

The `src/readability/readability.py` implementation starts with existing code,
then repeats:

```text
suggestion = critique(code)
code = fix(code, suggestion)
```

for five rounds. There is no execution gate in the loop. The paper evaluates
meaningful variable ratio, comments per line, and function units, with an LLM
helping judge meaningful variable names.

This is a good fit for post-pass cleanup but a bad fit for correctness:

```text
good:
    after candidate passes, ask Self-Refine to reduce naming confusion or
    simplify comments

bad:
    before candidate passes, ask Self-Refine to rewrite broad code for style
```

For DynaWorld, readability refinement should run only under strict file scope
and with tests after every refinement.

## Results

Main task set:

```text
Dialogue Response Generation
Code Optimization
Code Readability Improvement
Math Reasoning
Sentiment Reversal
Acronym Generation
Constrained Generation
```

Main table highlights:

```text
GPT-4 Code Optimization:
    27.3 -> 36.0

ChatGPT Code Readability:
    27.7 -> 63.1

GPT-4 Code Readability:
    27.4 -> 56.2

GPT-4 Dialogue Response:
    25.4 -> 74.6

GPT-4 Constrained Generation:
    15.0 -> 45.0

Math Reasoning:
    almost flat for all base models in the main setup
```

The math result matters. The authors attribute the small gains to inability to
identify whether there is any error. When the model thinks everything looks
good, self-refinement stalls.

Oracle-feedback math result:

```text
GPT-3.5 math:
    64.06 -> 68.9

ChatGPT math:
    74.8 -> 76.2

GPT-4 math:
    92.9 -> 93.8
```

Interpretation:

```text
external correctness information improves the loop
```

This supports the DynaWorld policy: model feedback should be grounded by
execution whenever correctness matters.

### Code Optimization Appendix

On PIE, the appendix table reports:

```text
Codex:
    13.1 % optimized

GPT-3.5:
    14.8

ChatGPT:
    22.2

GPT-4:
    27.3

Self-Refine with GPT-3.5:
    23.0

Self-Refine with ChatGPT:
    26.7

Self-Refine with GPT-4:
    36.0
```

The paper emphasizes that Self-Refine uses at most four samples, versus some
baselines using best-of-16 or best-of-32.

This is relevant to `codex exec` budget accounting:

```text
four sequential refine calls
vs
four independent samples
vs
one LLaMEA serial mutation chain of length four
```

The local runner should compare those fairly by Codex-call budget.

### Feedback Quality Ablation

The paper compares specific Self-Refine feedback, generic feedback, and no
feedback:

```text
Code Optimization:
    27.5 specific
    26.0 generic
    24.8 none

Sentiment Reversal:
    43.2 specific
    31.2 generic
    0 none

Acronym Generation:
    56.4 specific
    54.0 generic
    48.0 none
```

Lesson:

```text
the critique step is not ceremony
```

For DynaWorld, a reflection string must be scored by actionability. If it only
says "fix the bug" or "improve performance", it should not consume a repair
attempt.

### Iteration Curves

The paper reports diminishing returns:

```text
Code Optimization:
    y0 22.0 -> y1 27.0 -> y2 27.9 -> y3 28.8

Sentiment Reversal:
    y0 33.9 -> y1 34.9 -> y2 36.1 -> y3 36.8

Constrained Generation:
    y0 29.0 -> y1 40.3 -> y2 46.7 -> y3 49.7
```

It also notes that multi-aspect quality may not monotonically increase. A
candidate can improve one dimension and regress another.

Local implication:

```text
always keep per-stage metrics for every refinement
do not assume the last refinement is best
select by evaluator score, not iteration count
```

### Failure Analysis

The qualitative analysis of code optimization and math failures is highly
relevant:

```text
33 percent of unsuccessful cases:
    feedback pointed at the wrong location

61 percent of unsuccessful cases:
    feedback suggested an inappropriate fix

6 percent of unsuccessful cases:
    refiner failed to implement good feedback
```

In other words, most failures are feedback failures, not rewrite failures.

Local implication:

```text
reflection_builder quality is a first-class bottleneck
```

Do not over-invest in fancy candidate mutation before measuring whether the
failure feedback is actually correct and specific.

## Design Implications For `alpha_evolve`

### Use Self-Refine As A Helper, Not A Judge

Self-Refine is good when a target has subjective or multi-aspect quality:

```text
prompt clarity
test-suite coverage wording
JSON schema readability
failure explanation actionability
patch minimality
post-pass code readability
```

It is not sufficient for:

```text
functional correctness
runtime performance claims
renderer parity
data loader contract claims
baseline-beating claims
```

Those need execution and heldout checks.

### Add A Feedback Actionability Gate

Before a refinement attempt consumes budget, score the feedback:

```json
{
  "specific_location": true,
  "concrete_action": true,
  "grounded_in_visible_evidence": true,
  "mentions_unknown_hidden_gate": false,
  "risk_level": "low|medium|high"
}
```

If the feedback fails this gate, regenerate feedback or fall back to a
deterministic evaluator summary.

### Keep Candidate History But Select By Score

Self-Refine returns the final output, but the paper itself notes non-monotonic
quality. The local runner should store every iteration:

```text
attempt_id
parent_id
iteration
prompt
feedback
patch
visible score
gate results
diff stats
token cost
```

Then select the best visible candidate for hidden evaluation. Do not blindly
select the last iteration.

### Compare Against Independent Sampling

The paper compares Self-Refine to `k` independent outputs. That comparison is
mandatory locally:

```text
4 independent codex exec samples
1 initial + 3 self-refine retries
1 ReAct repair chain of length 4
1 LLaMEA serial chain of length 4
```

If Self-Refine does not beat independent samples on a problem class, use it only
for prompt polishing or feedback generation.

### External Feedback Upgrade

The math oracle-feedback result says the loop improves more when the model is
told the current output is wrong. In DynaWorld:

```text
unit test fail
smoke fail
benchmark regression
artifact mismatch
baseline row missing
schema violation
```

should all become explicit feedback facts before the model critiques the patch.

## Proposed Microlibs

### `self_refine_loop`

Responsibility:

```text
run a bounded generate/feedback/refine loop for a single candidate artifact
```

Inputs:

```text
task contract
initial artifact or generation prompt
feedback_builder
refine_prompt_builder
visible evaluator, optional
max_iters
selection_policy
```

Outputs:

```text
all iterations
selected candidate id
feedback quality scores
cost summary
```

### `feedback_actionability_scorer`

Responsibility:

```text
reject generic or ungrounded feedback before it is fed to Codex
```

Checks:

```text
specific file/function/metric/stage named
concrete action proposed
evidence source referenced
no hidden-gate leakage
no broad refactor request unless target allows it
```

### `refinement_history_selector`

Responsibility:

```text
choose the best iteration by evaluator metrics instead of final iteration
```

Selection modes:

```text
last:
    exact Self-Refine baseline

best_visible:
    choose best visible score among iterations

pareto:
    choose non-dominated candidate over score, diff size, runtime, and risk
```

### `soft_quality_refiner`

Responsibility:

```text
apply Self-Refine to non-authoritative soft outputs
```

Targets:

```text
prompt templates
failure summaries
generated tests before evaluator vetting
candidate notes
readability after hard gates pass
```

### `oracle_feedback_adapter`

Responsibility:

```text
convert deterministic evaluator facts into short feedback blocks for
Self-Refine-style refinement
```

This is the safest version for code:

```text
model feedback + evaluator facts -> refined patch
```

not:

```text
model feedback alone -> refined patch
```

## Local Falsification Tests

### Test 1: Specific Feedback Beats Generic Feedback

Setup:

```text
same failed candidate
same Codex model/profile
same max attempts
```

Compare:

```text
A: evaluator-grounded specific feedback
B: generic feedback, "improve this"
C: no feedback, retry from task only
```

Expected:
    A wins on solve rate or token cost.

Failure:
    No measurable difference. Then the feedback builder is not adding value.

### Test 2: Last Iteration Is Not Always Best

Procedure:

```text
run self_refine_loop on N deterministic tasks
score every iteration
compare final iteration with best_visible selection
```

Expected:
    Some tasks show non-monotonic score, so selector matters.

Failure:
    If final is always best, simpler exact Self-Refine is enough for that class.

### Test 3: Bad Feedback Is The Bottleneck

Procedure:

```text
sample failed refinements
label failure as:
    wrong location
    wrong fix
    good feedback, bad implementation
    evaluator ambiguity
```

Expected:
    If this paper transfers, most failures are wrong-location or wrong-fix
    feedback failures.

Action:
    Improve `feedback_actionability_scorer` before improving mutation.

### Test 4: External Feedback Helps

Compare:

```text
self feedback only
self feedback plus deterministic evaluator facts
self feedback plus full raw logs
```

Expected:
    Deterministic facts beat self feedback only and may beat raw logs by being
    smaller and clearer.

### Test 5: Code Readability Refine Cannot Break Passing Code

Procedure:

```text
take a passing microlib
run soft_quality_refiner on readability
run full gate after every iteration
```

Expected:
    Any iteration that breaks tests is rejected, and selector can choose an
    earlier passing iteration.

Failure:
    Style refinement silently changes behavior.

## Target Problems In This Repo

Good initial targets:

```text
alpha_evolve prompt template improvement
failure summary generation
generated-test phrasing and coverage
candidate metadata schema descriptions
post-pass readability for isolated microlibs
```

Risky targets:

```text
renderer kernel optimization
training loop behavior
data contract logic
baseline acceptance
anything with hidden shape/runtime dependencies
```

For risky targets, use Self-Refine only around evaluator feedback:

```text
run gate -> summarize facts -> critique -> codex exec patch -> run gate again
```

## How It Changes The AlphaEvolve Plan

Before this note:

```text
ReAct repair loop
Reflexion memory
LLaMEA serial evolution
```

After this note:

```text
ReAct repair loop
Self-Refine feedback/refine as a local retry primitive
Reflexion memory stores only successful/failed lessons across attempts
LLaMEA serial evolution uses the best refinement candidate, not necessarily the last
```

Self-Refine also adds one required metric to every experiment:

```text
feedback_quality
```

Candidate success is not just patch quality. It also depends on whether the
feedback was specific, actionable, grounded, and not stale.

## Open Questions

- Should `feedback_actionability_scorer` be rule-based at first, or another
  Codex/LLM critique?
- What is the cheapest model/profile that can write useful feedback for this
  repo without wasting Codex budget?
- Should the runner allow the same Codex call to critique and patch, or force
  two separate calls for inspectability?
- Can generated feedback be cached across sibling candidates, or is that too
  likely to spread wrong assumptions?
- How should soft-quality refinement interact with AGENTS.md anti-pattern rules,
  such as reducing LOC and avoiding config alias churn?
- Does Self-Refine help microlib prompt templates more than it helps code?

## Bottom Line

Self-Refine is a cheap local loop for improving one artifact. It is not a
program database, not evolution, and not a reliable correctness judge. The
transferable piece is the prompt discipline: feedback must be specific and
actionable, refinements should be bounded, every iteration should be retained,
and selection should be based on measured evidence rather than "latest draft."
