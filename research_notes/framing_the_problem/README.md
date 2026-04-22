# Framing the Problem

Alternate framings of the core DynaWorld problem: *novel-view synthesis
from self-supervised single-camera video, with optional multi-view /
synthetic budgets used as support when available*.

Each framing attacks the same underlying problem through a different
conceptual lens. They are not competing theories — they are different
languages for describing the same object. Some losses are easier to derive
in one framing; some contract violations are easier to spot in another.

## Current framings

- **`framing_1.md` — Information-Systems View.** Models the architecture
  as a channel. World tokens W should be a sufficient statistic under a
  group action (SE(3) × R). Leakage is I(W; C_true). Each training loss
  bounds a specific information-theoretic quantity. Introduces the "fixed
  point of optimization" question (a/b/c/d outcomes). Useful for
  deriving losses and reasoning about what a training signal actually
  enforces.

- **`framing_2.md` — Self-Sufficient World Tokens / Generative
  Reconstruction Contract.** Contract-based vocabulary: world tokens are
  "self-sufficient" iff W alone (+ query) suffices to render. Introduces
  the distinction between "decoding expansion" (VAE-like) and "generative
  reconstruction" (what a world model must do). Defines constraints C1–C9
  with concrete implementation checks. Introduces "frame-local state" as
  a named failure mode. Useful for implementation review and architecture
  auditing. Preserved as derivation history; not the current default.

- **`framing_3.md` — Bitter-Lesson Predictive Quotient (default
  baseline).** Patched, not replaced. One encoder, one exported world
  asset, time-conditioned scene generator, fixed rasterizer, held-out
  omitted-observation supervision under `D_var`, and rate/minimality as
  the representative selector. Formalizes a predictive quotient on
  supported queries: agreement is behavioral modulo gauge, not token
  equality or latent purity. Collapses framing_2's dedicated agreement /
  cross-view / adversarial mechanisms to diagnostics-only unless an
  escape hatch is admitted by measurement. This is the framing to start
  with when proposing anything new.

## How the framings relate

Same object, different language.

- Information-theoretically, self-sufficiency is: W is a sufficient
  statistic for the conditional rendering distribution,
  `P(image | W, C_q, t_q) = P(image | scene_true, C_q, t_q)`.
- Information-theoretically, generative reconstruction means W contains
  strictly more information than its encoder input V when V is a
  restricted view of the scene — which requires a learned generative
  prior.
- Information-theoretically, frame-local state is a hidden side channel
  that violates the I(image | W, C_q, t_q) sufficiency claim.

Framing 1 is a derivation framework; framing 2 is an audit framework;
framing 3 is the default baseline and evaluation framework. Keep all
three — different purposes, not competing answers.

For implementation, use `../training_contract_v1.md` as the operational
contract for the patched framing 3 baseline: sampler, signatures, losses,
diagnostics, escape hatches, support assumptions, export contract, and
failure tripwires.

**Default routing for a new question:**

- Proposing a new loss / mechanism / head -> framing 3 evaluation checklist.
- Implementing training, diagnostics, or export -> `../training_contract_v1.md`.
- Deriving what a proposed loss bounds -> framing 1.
- Checking whether an implementation leaks frame-local state -> framing 2
  constraints C1, C3, C5, C6, C7.
- Writing a paper-shaped argument -> whichever framing the argument is
  about; often framing 1 for bounds, framing 2 for contracts, framing 3
  for minimality.

## When to write a framing_N.md

Add a new framing when:

- A different conceptual language unlocks a design question the current
  framings cannot answer cleanly (e.g., a gauge-theoretic derivation, a
  category-theoretic composition story, a statistical-mechanics fixed-
  point analysis).
- An existing framing has a known gap or a contested claim that a fresh
  angle would resolve.
- A new framing would make a specific class of implementation mistakes
  easy to catch.

Do not add framings that are paraphrases of existing ones. The point is
coverage, not proliferation.

Do not mint `framing_4.md` for the predictive-quotient correction. That
correction lives in framing 3.

## Relationship to the three architectures

The three architectures in `../three_architectures_for_novel_view_synthesis.md`
(A: dimensional routing / B: adversarial / C: multi-camera consistency)
map cleanly to the earlier framings, and should now be evaluated through
framing 3 before shipping:

- In framing 1: A has no formal guarantee on I(W; C); B directly bounds
  I(W; C) via variational mutual information; C enforces structural
  C-equivariance.
- In framing 2: A makes a weak routing hope toward C1 (self-sufficiency);
  B provides a non-structural probe for C1; C enforces C2 (4D
  consistency) directly via the training regime.
- In framing 3: A/B/C are mechanisms. They ship only if they satisfy the
  predictive-quotient checklist or appear as diagnostics / admitted
  escape hatches in `../training_contract_v1.md`.

If the architectures doc ever contradicts a framing, update the doc.
Framings describe the target; architectures are attempts to hit it.
