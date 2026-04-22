# How the prompt and guidance could have been better for model architecture research

## Context and goal

This note is an operating guide for future DynaWorld prompts to external
reasoning models. It reflects on a recent ChatGPT Pro steering prompt and the
scientist response around the stochastic interventional quotient model,
Framing 3, and whether a Framing 3.5/4 was needed.

The lesson is not "ask for more detail." The useful move was narrower:
turn model-architecture research into adjudication under a contract. Ask the
model to classify claims, state support conditions, prove or weaken each
claim, construct counterexamples, and only then recommend whether a mechanism,
escape hatch, training contract, or new framing is warranted.

The scientist response was useful because it did not merely generate another
architecture. It corrected an overclaim in Framing 3, introduced the predictive
quotient on supported queries, separated sufficiency from minimality, gave a
cancellation counterexample, fixed the camera/time boundary, and kept v1
deterministic while admitting posterior semantics. That is the output shape to
reward.

## What the prompt got right

The strongest part of the prompt was that it asked for adjudication, not
ideation.

- It asked whether the stochastic interventional quotient model was a
  correction, formalization, or escape hatch. That forced the answer to place
  the proposal in the maintenance system instead of treating it as a shiny new
  mechanism.
- It required subsumption to be tested per invariant. That is better than
  asking "does Framing 3 hold?" because a framing can be right in its
  recommended baseline while overclaiming one theorem.
- It asked for decisions on minimality and stochasticity. That separated
  "W is sufficient for supported predictive queries" from "W is a minimal
  latent" and from "the implementation must sample a posterior."
- It forced the camera/time boundary. That prevented the response from hiding
  ambiguity in whether camera is an encoder output, decoder query, user input,
  predicted nuisance variable, or latent side channel.
- It requested a cancellation counterexample. That made the critique falsify a
  specific claim instead of vaguely warning that leakage might happen.
- It asked for a corrected training contract and a decision on whether to
  create Framing 4. That converted critique into a repo-maintenance action.

Reusable principle:

> A good architecture-research prompt names the decisions that must be made,
> then asks the model to classify evidence against each decision. It does not
> ask the model to "think of an architecture" first.

## Where the prompt was under-specified

### It was too leading toward Framing 3.5/4

The prompt made "maybe we need Framing 3.5/4" feel like the expected move.
Future prompts should make a new framing pay a burden of proof:

- What existing claim cannot be repaired in place?
- What class of mistakes would the new framing catch that the existing
  framings miss?
- Which existing framing becomes simpler or less misleading after the split?
- What would be lost by not creating it?

Default answer should be "patch the current framing" unless the model can name
a distinct conceptual language with a distinct audit value.

### Support conditions were not forced before recommendations

The response improved the project by adding "supported queries." The prompt
should have required that upfront.

Every theorem-like claim needs explicit support conditions before any
recommendation:

- observation-budget distribution
- query distribution
- camera/time variables and which are observed, predicted, or user-supplied
- model capacity and optimizer assumptions
- renderer differentiability and identifiability assumptions
- whether the claim is on-support, off-support, or posterior-semantic

Without this, a model will silently turn a support-limited predictive claim
into a universal world-model claim.

### Theorem, taste, and implementation were not separated

The prompt should have required every important statement to be tagged:

- **Theorem-like:** true under stated assumptions; should have proof
  obligations and counterexamples.
- **Definition:** a naming choice that changes what future claims mean.
- **Taste:** a research preference, such as "deterministic v1 is simpler."
- **Implementation:** an engineering decision, such as not adding a stochastic
  latent path yet.
- **Escape hatch:** not in the v1 contract; admitted only after a trigger.

The scientist response implicitly did this well: predictive quotient was a
theorem-like correction, deterministic v1 was an implementation/taste decision,
and posterior semantics were an interpretation rather than a required sampling
mechanism. Future prompts should make that separation explicit.

### Mechanism overcommitment was not penalized hard enough

"Stochastic interventional quotient" sounds like it wants to become a model
component. That is dangerous. In this project, the first question for any
mechanism-shaped noun is:

> Does this enter the gradient path, or is it only a semantic correction to the
> training contract?

If it enters the gradient path, the prompt must require a predictive-coverage
failure: name the invariant the baseline predictive objective plus
rate/minimality cannot reach, the support where it cannot reach it, and the
measurable trigger that admits the mechanism. Otherwise the mechanism should
be downgraded to vocabulary, diagnostic, or future escape hatch.

### Proof obligations were not attached to each claim

The prompt asked for a counterexample, which worked, but it did not require a
claim ledger. Future prompts should force every consequential claim to include:

- what would prove it
- what would weaken it
- what counterexample would kill it
- what edit to the repo would follow

This matters because model responses often contain one useful correction mixed
with three plausible but unsupported recommendations. The prompt should make
unsupported recommendations easy to quarantine.

## How to structure future architecture-research prompts

Use a two-pass structure.

### Pass 1: adjudication before design

Ask the external model to operate on claims, not architectures.

1. Decompose the contested proposal into atomic claims.
2. Define variables, support, losses, query distribution, and boundaries.
3. Tag each claim as theorem-like, definition, taste, implementation, or escape
   hatch.
4. Audit each claim against the current baseline.
5. Produce counterexamples for overclaims.
6. State the smallest correction that preserves the baseline if possible.

Only after this pass should the model discuss new mechanisms.

### Pass 2: mechanism budget

If the model recommends a mechanism, require it to justify the budget:

- Which invariant is not covered by the baseline predictive objective,
  rate/minimality, or export contract?
- Is the gap on-support or off-support?
- Is the mechanism a training loss, architecture change, data-budget change,
  diagnostic probe, post-training step, or semantic vocabulary?
- What is the retirement condition?
- What simpler edit would fail, and why?

The null recommendation must be allowed: "no new mechanism; patch the theorem
and keep deterministic v1" is a valid research output.

## Required output fields and proof obligations

Future prompts should require these fields.

### Claim ledger

For each claim:

- `claim_id`
- `statement`
- `tag`: theorem-like / definition / taste / implementation / escape hatch
- `status`: true / false / conditional / unsupported / not needed
- `support_conditions`
- `proof_obligation`
- `counterexample_or_failure_case`
- `repo_action`

The `repo_action` must be concrete: patch a sentence, add a caveat, demote to
diagnostic, keep v1 unchanged, create a new training contract, or create a new
framing.

### Predictive-coverage audit

For each invariant:

- invariant name
- observable consequence
- query/camera/time support where the consequence is observable
- whether the current predictive objective reaches it
- what support-limited behavioral-equivalence argument is being used
- where the argument fails
- whether the failure admits a mechanism or only a caveat

This would have made the Framing 3 correction cleaner: the predictive
objective supports a quotient on supported queries, not a universal claim
about all latent scene truth.

### Minimality and sufficiency audit

Require the model to answer separately:

- Is W sufficient for the supported rendering distribution?
- Is W minimal among sufficient statistics?
- Is minimality required for the training contract?
- Does the proposed model need to identify true scene variables, or only the
  predictive quotient induced by the loss?

Do not let "sufficient" silently become "minimal" or "true."

### Stochasticity audit

Require the model to distinguish:

- posterior uncertainty in the semantics
- stochastic sampling in the implementation
- noise used as augmentation
- residual generative prior as an escape hatch
- deterministic approximation to a conditional mean or mode

Then ask whether v1 changes. In the recent episode, the right answer was:
admit posterior semantics, keep deterministic v1 unless a measured failure
requires a stochastic residual path.

### Boundary audit

Require exact roles for:

- observed video frames
- held-out target frames
- predicted cameras
- user-supplied query cameras
- time index or time query
- latent world representation W
- any stochastic variable Z
- renderer/decoder inputs and outputs

This catches camera/time equivocation before it enters the architecture.

### Counterexample section

Require at least one constructive counterexample for every theorem-like
overclaim. The useful pattern from this episode was cancellation:

- W carries a forbidden dependency.
- Another variable carries the compensating inverse dependency.
- The rendered output still matches on the supervised support.
- Therefore the original claim was too strong unless additional assumptions
  exclude the cancellation.

The counterexample does not need to be likely. It needs to show which theorem
is not proven by the loss.

## Anti-patterns to ban

- **Framing 3.5/4 magnet:** treating a new name as progress before proving the
  existing framing cannot be patched.
- **Mechanism noun laundering:** introducing a technical-sounding object and
  letting it drift into the gradient path without a subsumption failure.
- **Universal theorem from support-limited loss:** claiming true scene recovery
  when the loss only identifies predictive equivalence on supported queries.
- **Sufficiency/minimality collapse:** using "sufficient statistic" and
  "minimal latent" interchangeably.
- **Stochasticity conflation:** treating posterior semantics, sampled
  implementation, denoising noise, and residual generative priors as the same
  thing.
- **Camera boundary equivocation:** switching between predicted camera,
  training camera, and user query camera without saying which one enters the
  renderer.
- **Taste disguised as proof:** recommending deterministic or stochastic v1
  without saying whether the reason is theorem, implementation cost, or
  research taste.
- **Mechanism-first rescue:** responding to an overclaim by adding losses,
  heads, or stages before trying the smaller correction: weaken the theorem.
- **Architecture answer to a contract question:** proposing components when
  the user asked what the representation means under the loss.

## Reusable prompt skeleton

Use this when asking an external LLM to evaluate a proposed model-architecture
idea or theoretical correction.

```text
You are helping adjudicate a model-architecture research claim for DynaWorld.
Do not ideate new architectures until the adjudication pass is complete.

CURRENT BASELINE:
<paste the current framing/training contract. State which doc is default.>

CONTESTED PROPOSAL:
<paste the new idea, e.g. stochastic interventional quotient model.>

QUESTION:
Classify whether the proposal is:
- a correction to an overclaim in the baseline
- a formalization of what the baseline already meant
- an escape hatch admitted only after a trigger
- a new mechanism that must enter the training contract
- a genuinely new framing
- unnecessary

REQUIRED METHOD:
1. Decompose the proposal into atomic claims.
2. Before recommendations, define support conditions:
   - observation-budget distribution
   - query/camera/time support
   - observed, predicted, and user-supplied variables
   - model capacity / optimizer assumptions
   - renderer and decoder assumptions
3. Tag each claim:
   theorem-like / definition / taste / implementation / escape hatch.
4. For each theorem-like claim, provide:
   - proof obligation
   - support conditions
   - counterexample or failure case
   - whether the claim is on-support or off-support
5. Run a predictive-coverage audit per invariant:
   - invariant
   - observable consequence
   - whether the baseline predictive objective reaches it
   - where supported-query behavioral equivalence holds
   - where it fails
6. Separate sufficiency, minimality, and stochasticity:
   - Is W sufficient for supported predictive queries?
   - Is W minimal? Is minimality required?
   - Is stochasticity semantic, implementation-level, or an escape hatch?
7. Decide the smallest repo action:
   - patch existing framing
   - patch training contract
   - demote mechanism to diagnostic
   - add escape hatch and trigger
   - create new framing
   - no change

FORBIDDEN MOVES:
- Do not create Framing 4 unless you prove patching the current framing is
  insufficient.
- Do not recommend a new loss/head/stage unless you identify an invariant not
  covered by the baseline predictive objective, rate/minimality, or export
  contract.
- Do not use "sufficient", "minimal", "true scene", and "predictive quotient"
  interchangeably.
- Do not conflate posterior semantics with a required stochastic implementation.
- Do not cite papers unless the citation is verified.

OUTPUT FORMAT:

<adjudication>
  <support_conditions>...</support_conditions>
  <claim_ledger>
    <claim id="C1">
      <statement>...</statement>
      <tag>theorem-like | definition | taste | implementation | escape_hatch</tag>
      <status>true | false | conditional | unsupported | not_needed</status>
      <support_conditions>...</support_conditions>
      <proof_obligation>...</proof_obligation>
      <counterexample_or_failure_case>...</counterexample_or_failure_case>
      <repo_action>...</repo_action>
    </claim>
  </claim_ledger>
  <subsumption_audit>...</subsumption_audit>
  <minimality_vs_sufficiency>...</minimality_vs_sufficiency>
  <stochasticity_audit>...</stochasticity_audit>
  <camera_time_boundary>...</camera_time_boundary>
  <smallest_correction>...</smallest_correction>
  <mechanism_budget>
    If you recommend any new mechanism, justify why the baseline loss does not
    subsume the invariant and give a trigger plus retirement condition.
  </mechanism_budget>
  <new_framing_decision>
    Yes/No. If Yes, state the distinct conceptual language and the class of
    mistakes it catches. If No, state the exact patch to the existing framing.
  </new_framing_decision>
  <open_questions>...</open_questions>
</adjudication>
```

## Open questions and falsification checks for prompt quality

After receiving an external answer, audit the prompt by asking:

- Did the answer change a concrete repo decision, or only add vocabulary?
- Did it state support conditions before recommending changes?
- Did it weaken at least one overstrong claim if the baseline deserved it?
- Did it preserve the baseline when a smaller correction was enough?
- Did it distinguish theorem-like claims from taste and implementation?
- Did it produce a counterexample that would have been hard to see from the
  original framing?
- Did it say "no new framing" when no distinct conceptual language was needed?
- Did it identify which claims remain unsupported?
- Could the same answer be pasted into an unrelated project with only names
  changed? If yes, the prompt failed.

A useful prompt makes the model slower to add mechanisms and faster to expose
the exact claim that was too strong.
