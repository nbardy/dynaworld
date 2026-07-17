# 008 - ReAct: Synergizing Reasoning and Acting in Language Models

Status:
    first-pass detailed note

Primary sources:
    https://arxiv.org/abs/2210.03629
    https://arxiv.org/pdf/2210.03629
    https://react-lm.github.io/
    https://github.com/ysymyth/ReAct

Bibliographic metadata:
    Authors: Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran,
    Karthik Narasimhan, Yuan Cao.
    First arXiv submission: 2022-10-06.
    Latest arXiv version inspected: v3, 2023-03-10.
    Venue: ICLR 2023.

Why this paper matters for alpha_evolve:
    ReAct is the minimal agent loop under Voyager, SWE-agent, OpenHands, and many
    tool-using coding agents. It does not have evolution, candidate databases, or
    a skill library. Its contribution is the interleaved protocol:
    thought, action, observation. For DynaWorld, ReAct is the simplest baseline
    for `codex exec` loops before adding archives, islands, curricula, or
    verified skill libraries.

One-sentence mechanism:
    Prompt a language model with examples where it alternates between reasoning
    traces, environment/tool actions, and observations, so external feedback
    grounds future reasoning and reasoning guides future actions.

## Reading Questions

- What is the executable feedback signal?
  Environment observations from tools or simulators. In the paper these are
  Wikipedia API responses, ALFWorld text observations, and WebShop page states.

- What is being searched: code, trajectories, tests, prompts, policies, or
  memories?
  Task-solving trajectories. The model writes thoughts and actions online, not a
  persistent program database.

- What is the population/database/selection mechanism?
  None. ReAct is a prompt protocol, not an evolutionary method. The only state is
  the trajectory context accumulated inside the current episode.

- What evidence proves the loop improves over one-shot generation?
  ReAct is compared against standard prompting, chain-of-thought, act-only
  prompting, imitation/RL baselines, and an Inner-Monologue-style ablation. It
  improves factual grounding and interactive decision-making success, but it is
  not itself an optimizer over repeated candidates.

- What does the method assume that DynaWorld does not have?
  It assumes the environment action space is safe and bounded. A repo-level
  coding agent can mutate files, run commands, and affect artifacts, so local
  ReAct-style action spaces must be heavily sandboxed and evaluator-gated.

## Mechanism

ReAct augments a normal action space with a language thought space:

```text
ordinary action: affects environment
thought action: affects only the agent context
observation: returned by environment after ordinary action
```

The trajectory format is:

```text
Thought 1: ...
Action 1: ...
Observation 1: ...
Thought 2: ...
Action 2: ...
Observation 2: ...
...
```

In knowledge tasks, thoughts and actions alternate densely. In long-horizon
decision tasks, thoughts can be sparse: the model decides when to think and when
to act.

The key synergy:

- reasoning helps decide what to search, inspect, click, or execute next;
- action retrieves grounding information not present in model weights;
- observations update the reasoning state;
- the trajectory remains inspectable and editable by humans.

This is much smaller than Voyager. There is no automatic curriculum, no
long-term skill library, no candidate archive, and no hard promotion step. That
is why it is a good baseline.

### Knowledge Tasks

For HotpotQA and FEVER, ReAct uses a small Wikipedia API with:

```text
search[entity]
lookup[string]
finish[answer]
```

This action space is intentionally weaker than modern retrieval. It returns a
small amount of page text and forces the model to reason about what to search or
lookup next.

Prompt examples include thoughts that:

- decompose a multi-hop question;
- extract relevant facts from observations;
- reformulate searches when a page is missing;
- distinguish internal belief from retrieved text;
- synthesize the final answer.

This maps cleanly to repo work:

```text
search code -> inspect result -> reason about contract -> run command -> inspect logs
```

But the local action space is more dangerous. File edits and shell commands need
path, timeout, and evaluator boundaries.

### Decision Tasks

For ALFWorld and WebShop, ReAct uses sparse thoughts. The model may think to
choose a likely search location, decide when a subgoal is complete, or bridge a
noisy product page to a user instruction.

This matters because too much reflection can become noise. The paper's
ReAct-IM ablation, which restricts thought to dense external-feedback style,
underperforms ReAct in ALFWorld. The useful thoughts are not just restatements
of observations. They perform high-level goal decomposition, commonsense search,
progress tracking, and exception handling.

For `alpha_evolve`, reflection should not become a verbose echo of logs. It
should answer:

```text
What changed?
What evidence did the evaluator give?
What next action follows from that evidence?
What assumption might now be wrong?
```

### Human Thought Editing

The paper shows that humans can edit a couple of thoughts in a failed ALFWorld
trajectory and change downstream behavior. This is a practical collaboration
point: thoughts are a control surface.

For DynaWorld, a human reflection like:

```text
The candidate is improving smoke metrics by making the task easier, not by
fixing the renderer path. Inspect the data contract before another patch.
```

can be more useful than editing code directly. The runner should store human
thought edits as candidate reflection, not as evaluator truth.

## Evaluation

Benchmarks:

- HotpotQA: multi-hop question answering.
- FEVER: fact verification.
- ALFWorld: text household environment with six task types.
- WebShop: web shopping environment with real product text and structured
  options.

Baselines:

- Standard prompting.
- Chain-of-thought.
- Chain-of-thought with self-consistency.
- Act-only prompting.
- ReAct combined with CoT-SC fallback heuristics.
- Imitation/RL baselines for ALFWorld and WebShop.
- Inner-Monologue-style prompting in ALFWorld.
- Fine-tuned variants in the paper's HotpotQA study.

Important results:

- ReAct beats Act on HotpotQA and FEVER, showing reasoning helps action.
- ReAct beats CoT on FEVER and is slightly behind CoT on HotpotQA, showing
  acting helps factual grounding but can constrain reasoning.
- ReAct plus CoT-SC performs best in the two knowledge tasks, showing internal
  knowledge and external tool grounding should be combined rather than treated
  as mutually exclusive.
- On ALFWorld, the best ReAct prompt reaches 71 percent success versus 45
  percent for Act and 37 percent for BUTLER.
- On WebShop, ReAct reaches 40 percent success versus 30.1 percent for Act and
  29.1/28.7 percent for the IL/IL+RL baselines reported in the table.
- ReAct-IM underperforms ReAct on ALFWorld, supporting sparse high-level
  reasoning over dense observation echoing.

Failure analysis:

- CoT has more hallucinated facts.
- ReAct is more grounded, but can suffer reasoning errors caused by the rigid
  thought/action structure.
- ReAct can loop by repeating prior thoughts and actions.
- Non-informative search results can derail the trajectory.
- Tool quality matters.

The paper also reports that ReAct trajectories are a strong fine-tuning format.
With 3,000 generated correct-answer trajectories, smaller fine-tuned models can
outperform larger prompted models in the HotpotQA setup. That is interesting for
future local trace distillation, but not a first-version requirement.

## Failure Modes

### Tool Retrieval Can Mislead The Reasoner

ReAct grounds reasoning in observations, but if the action retrieves irrelevant
information, the model can reason from bad evidence.

DynaWorld rule:

```text
Observations need provenance and stage labels.
Do not let a failed grep, stale log, or partial smoke output become unqualified evidence.
```

### Reasoning/Action Loops

The paper explicitly observes repetitive thought/action loops. Repo agents can
loop over the same failing command or same patch idea.

DynaWorld rule:

```text
Track repeated actions and repeated failure signatures. Stop or change operator
after a loop threshold.
```

### Thoughts Are Useful But Not Evidence

Thoughts improve interpretability and action selection, but they are not proof.
This is the same lesson as EoH candidate thoughts and Eureka reflection.

DynaWorld rule:

```text
Store thoughts and reflections, but selection depends on evaluator artifacts.
```

### Action Space Safety

The paper's ethics section notes that tool-using language models can take
harmful actions if the action space is unsafe. Their experiments avoid real
buying or private-information actions.

DynaWorld rule:

```text
The local action space for evolution should be candidate worktrees, allowed
paths, bounded commands, and immutable evaluators.
```

### Too Much Log Echo Is Worse Than Sparse Reasoning

ReAct-IM shows that dense external-feedback style is not enough. It lacks
subgoal-completion reasoning and commonsense planning.

DynaWorld rule:

```text
Prompt feedback should be compressed into decisions, not pasted as raw logs.
```

## DynaWorld Mapping

### Baseline: `react_repair_loop`

Before building LLaMEA/CodeEvolve-style evolution, implement a ReAct-shaped
repair loop:

```text
Thought: explain the next repo action from current evidence.
Action: run one allowed shell/read/eval/edit action.
Observation: capture output, failure, or evaluator result.
```

For `codex exec`, this may be encoded as prompt sections rather than live tool
calls:

```text
Current evidence
Allowed actions
Previous attempts
Requested next patch
Required observation format
```

The baseline test is:

```text
Can a ReAct-style bounded repair loop solve the microlib without archive
selection or evolution?
```

If yes, evolution must beat that baseline, not only one-shot prompting.

### Candidate Trace Schema

ReAct suggests a simple trace format to store for every candidate:

```json
{
  "candidate_id": "...",
  "trace": [
    {
      "kind": "thought",
      "text": "The F32 smoke failed because the colorize path was not exercised."
    },
    {
      "kind": "action",
      "tool": "apply_patch",
      "summary": "Patch helper signature."
    },
    {
      "kind": "observation",
      "stage": "smoke_f32",
      "result": "failed",
      "log_path": "logs/..."
    }
  ]
}
```

This is not a replacement for score metrics. It is an audit trail.

### Loop Detection

Add a small loop detector to the runner:

```text
same command repeated N times
same exception repeated N times
same file/patch target repeated without score movement
same thought/operator repeated after failure
```

If detected, the runner should switch operator, retrieve a different skill, or
mark the candidate failed.

### CoT-SC Plus ReAct Analogy

The paper's best knowledge-task results combine internal reasoning with external
tool use. Local analogy:

```text
internal samples:
  multiple Codex patch proposals or reasoning sketches

external grounding:
  hard evaluator, grep, tests, smoke, baseline docs

fallback:
  if internal samples disagree, run grounding before editing
  if grounded repair stalls, sample a fresh reasoning/patch route
```

This argues against a single rigid loop. Some problems need more sampling before
acting; others need more tool grounding before sampling.

### Human Thought Editing

Store user steering as a first-class reflection artifact:

```text
human_reflection.md
applies_to_candidate_id
editable_summary
forbidden_shortcut
next_focus
```

This keeps human insight available to future prompts without converting it into
hard evaluator truth.

## Falsification Tests

### Test 1: ReAct Baseline Before Evolution

Pick one small microlib. Compare:

```text
one-shot codex exec
ReAct-style repair loop with max 4 attempts
LLaMEA-style serial evolution with the same Codex-call budget
```

Claim falsified if evolution does not beat the simpler ReAct repair loop.

### Test 2: Thought Ablation

Run the repair loop with and without explicit reasoning/reflection fields.

Claim falsified if thoughts do not reduce repeated failures, improve action
selection, or make traces easier to debug.

### Test 3: Observation Provenance

Inject stale or partial logs into the prompt packer.

Claim falsified if the runner treats unqualified logs as current evaluator
evidence.

### Test 4: Loop Stopper

Create a fixture where the same error repeats after two repairs.

Claim falsified if the runner keeps applying the same action/operator after the
loop threshold.

### Test 5: Human Thought Edit

Add a human reflection correcting the candidate's mistaken assumption.

Claim falsified if the next prompt ignores the reflection or mutates the hidden
evaluator instead of changing the candidate approach.

## Notes For Future Papers

- Reflexion should be read as adding persistent verbal memory to this ReAct
  loop.
- Self-Refine should be compared against ReAct when no external environment
  action is available.
- LATS should be read as search over ReAct trajectories rather than one greedy
  trajectory.
- SWE-agent should be evaluated as a repo-specific ReAct interface with a better
  action language.

## Bottom Line

ReAct is the minimum viable agent loop:

```text
thought
action
observation
repeat
```

For DynaWorld, it should be the baseline repair loop under every heavier
evolution design. If a microlib cannot beat one-shot Codex with a simple ReAct
repair trace, adding islands, program databases, or skill retrieval will mostly
hide the weak evaluator rather than fix it.
