---
name: deep-critical-thought
description: Use when asked to write, extend, or continue deep technical thinking documents; when the user asks for more branches, backtracking, proofs, derivations, hypothesis challenges, red-team analysis, or agent notes that preserve rigorous exploratory reasoning.
---

# Deep Critical Thought

Use this skill when the user wants a durable thinking document, not a quick
answer. The goal is to create useful visible technical reasoning: assumptions,
derivations, hypotheses, counterarguments, tests, and revisions. Do not present
private hidden chain-of-thought; write a clear external engineering note that
future agents can audit.

## Default Output Location

For this repo, write raw thinking documents as loose notes:

```text
agent_notes/loose_notes/{YYYY-MM-DD_HH-MM-SS}_{topic_slug}.md
```

If extending an existing thought document, patch that document unless the user
asks for a separate note or the topic has clearly changed.

Use `agent_notes/key_learnings.md` only for compact surprising lessons that
change future reasoning. Do not turn it into a journal.

## Document Contract

A good thought document should include:

- **Context:** what triggered the note, what code/data/evidence exists.
- **Current model:** the best working theory, stated as provisional.
- **Assumptions:** each important premise made explicit.
- **Math and ranges:** units, bounds, invariants, dimensions, and failure
  thresholds.
- **Derivations:** enough algebra or geometry for future agents to verify the
  claim.
- **Branches:** plausible alternative explanations, not just the favored one.
- **Backtracks:** places where prior assumptions could be wrong or have already
  been weakened.
- **Falsification tests:** cheap experiments that can prove a hypothesis wrong.
- **Decision implications:** what to do next if each branch is supported or
  invalidated.
- **Open questions:** unresolved facts that block confidence.

Prefer dense, searchable headings. Use ASCII math and code blocks for formulas.

## Thinking Modes

### Branch

Use when the user says "branch", "more ideas", "what else", "alternatives", or
when one theory is becoming too dominant.

For each branch:

```text
Hypothesis:
    ...
Why it might be true:
    ...
What would make it false:
    ...
Cheap test:
    ...
If supported:
    ...
If invalidated:
    ...
```

Include uncomfortable branches: convention bugs, stale data, logging artifacts,
bad assumptions, and "the previous note is wrong."

### Backtrack

Use when the user asks to challenge assumptions, revisit old reasoning, or when
new evidence contradicts the note.

Do not delete the old hypothesis. Mark it:

```text
Status:
    supported / weakened / invalidated / unresolved
Evidence:
    ...
Replacement model:
    ...
```

Backtracking is not failure. It is the mechanism that keeps notes useful.

### Continue

When the user says "continue", do not merely summarize the existing note. Add a
new expansion pass that explores fresh angles.

Choose several of these axes:

- alternative representations
- exact proofs and derivations
- numerical ranges and stability margins
- implementation sketches
- diagnostic metrics
- synthetic tests
- real-data experiments
- red-team questions
- failure taxonomy
- invariants and schemas
- future architecture options
- ways the current theory could be wrong

Name the pass explicitly:

```text
## Expansion Pass N: Short Title
```

### More / Longer

When asked for "more" or "longer", increase depth, not filler. Add concrete
math, tests, branches, examples, tables, and counterexamples. Revisit earlier
sections and sharpen vague claims into falsifiable statements.

## Proof And Math Standards

For every mathematical claim, try to include:

- definitions of symbols
- units or coordinate frame
- assumptions that make the formula true
- edge cases where it breaks
- how to test it against code or data

Useful patterns:

```text
Claim:
    ...
Assumptions:
    ...
Derivation:
    ...
Failure case:
    ...
Diagnostic:
    ...
```

Prefer dimensionless ratios for cross-config comparisons:

```text
step_length / camera_radius
scene_radius / camera_radius
projected_radius / image_half_extent
moment_norm / scene_scale
loss / mean_color_loss
```

## Hypothesis Hygiene

Avoid writing as if one theory is settled unless the evidence is strong.

Use labels:

```text
Current belief:
    ...
Confidence:
    low / medium / high
Evidence:
    ...
Could be wrong if:
    ...
```

Separate:

```text
observed fact
inference from fact
speculation
proposal
```

If a user suggests an idea, treat it seriously enough to formalize and test,
even if it is not the current favored direction.

## Falsification Loop

Every major section should eventually point to a test:

```text
1. What would we measure?
2. What result supports the hypothesis?
3. What result weakens it?
4. What should change next?
```

Prefer cheap tests before architecture changes:

```text
synthetic unit test
one-step smoke
diagnostic print
quantile report
local artifact
ablation with one knob changed
```

## When Writing About Code

Ground the note in actual files when possible:

```text
File:
    src/...
Current behavior:
    ...
Relevant config:
    ...
```

Use `rg`, `sed`, and small reads to verify current behavior before writing
specific claims. If code was not checked, mark the statement as an assumption.

## General Red-Team Patterns

Before considering a thought document done, challenge it from several general
angles. Choose examples that fit the actual topic instead of copying a fixed
checklist.

### Evidence Quality

Ask whether the evidence really supports the claim:

```text
What exactly was observed?
What is inferred rather than observed?
Could the measurement, benchmark, plot, log, or visualization be misleading?
Is there a simpler explanation for the result?
What result would make us abandon this hypothesis?
```

### Boundary And Scale

Ask where the idea breaks:

```text
Does it work at small and large scale?
Does it work on edge cases, degenerate cases, and adversarial inputs?
Does it depend on a hidden constant, unit, coordinate frame, or distribution?
What happens when the main variable changes by 10x or 100x?
Is the proposed invariant actually invariant under the transformations we care about?
```

### Alternative Mechanisms

Force multiple explanations:

```text
What else could produce the same symptom?
Could this be a data issue instead of a model issue?
Could this be a measurement issue instead of a real improvement?
Could this be a numerical issue instead of a conceptual issue?
Could this be a capacity/optimization issue instead of an architecture issue?
```

### Cost And Tradeoff

Make consequences explicit:

```text
What complexity does this add?
What does this make harder to debug?
What assumptions become part of the system contract?
What existing baseline or use case might regress?
What is the cheapest experiment before committing to the design?
```

## Domain Example Prompts

Use these as examples, not as mandatory checklists.

### Model Architecture

```text
Is the architecture solving the intended problem or adding a bypass?
What symmetry or invariance does this encode?
What useful behavior does the architecture forbid?
Could a smaller baseline or simpler parameterization explain the same result?
Does the change improve generalization or only increase memorization capacity?
What ablation isolates the new architectural idea from parameter count?
```

### Training And Optimization

```text
Is the failure before or after the optimizer step?
Could LR, schedule, initialization, clipping, or parameter grouping explain it?
Are gradients absent, noisy, saturated, exploding, or flowing to the wrong module?
Does the loss beat trivial baselines such as mean, copy, blur, or heuristic outputs?
Could the metric improve while the learned representation gets worse?
What one-step or few-step smoke test separates numerics from learning dynamics?
```

### CUDA / Kernel / Systems Optimization

```text
Are we measuring wall time, kernel time, throughput, latency, or memory bandwidth?
Did batch size, problem size, precision, logging, synchronization, or warmup change?
Is the workload launch-bound, compute-bound, memory-bound, or occupancy-bound?
Are there hidden sync points, host-device transfers, atomics, or non-coalesced accesses?
Does the optimization preserve numerical results within an acceptable tolerance?
What roofline or microbenchmark would falsify the bottleneck theory?
```

### CV / Rendering / Geometry Math

```text
What coordinate frame and units does every tensor live in?
Which convention is being assumed: row/column vectors, world-to-camera, camera-to-world?
What happens near singularities: zero depth, tiny determinant, parallel rays, extreme FoV?
Are transforms, scales, covariances, and rays normalized consistently?
Does the derivation preserve projection under similarity transforms?
Can a synthetic scene with known ground truth reproduce or falsify the claim?
```

### Data / Pipeline Bugs

```text
Could stale artifacts, wrong splits, wrong preprocessing, or mislabeled files explain it?
Is the config loading the intended data and schema version?
Are source commands, timestamps, checksums, and output directories recorded?
Does a tiny known input produce the expected output through the whole pipeline?
Could caching or previous generated output mask the real behavior?
```

### Research Claims

```text
What is the narrow claim and what is the broad claim?
Which evidence supports each one?
What baseline would make the result unimpressive?
What confounder would explain the observed gain?
What prediction does this theory make on a different dataset, scale, or regime?
What would make us downgrade this from conclusion to speculation?
```

### API / Product / UX Reasoning

```text
What user goal is this optimizing for?
Could the same behavior surprise users in another workflow?
What state, permission, latency, or failure mode is being hidden?
Does the interface make the safe path easy and the dangerous path explicit?
What minimal prototype or instrumentation would reveal whether the assumption is true?
```

## Final Response To User

Keep the final response short. Say what file was created or updated, what major
sections were added, and whether validation such as `git diff --check` passed.
