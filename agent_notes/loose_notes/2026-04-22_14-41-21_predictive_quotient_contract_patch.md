# Predictive quotient contract patch

Session on 2026-04-22. User asked to use two subagents:

1. Patch the architecture docs from the latest ChatGPT Pro / scientist
   verdict.
2. Write a meta-philosophy note on how the prompt and guidance could have
   been better for model architecture research.

## What changed

- Patched `research_notes/framing_the_problem/framing_3.md` from
  "bitter-lesson subsumption" to "bitter-lesson predictive quotient."
  The important correction: held-out prediction identifies behavior on
  supported queries, not latent purity, token equality, or absence of
  cancellation.
- Added `research_notes/training_contract_v1.md` as the operational
  contract: `D_var` sampler, model signatures, `L_pred + beta * Rate(W0)`,
  diagnostics, escape hatches, support assumptions, export contract, and
  tripwires.
- Added
  `research_notes/meta_philosophy/how_prompt_guidance_could_have_been_better_for_model_architecture_research.md`
  as a reusable guide for future external-LLM architecture prompts.
- Updated routing docs and stale language from F1-F6 to F1-F7 where it was
  clearly part of the current strategic guide.

## Integration fixes after worker patches

- Tightened the sampler overlap rule. The first worker wrote "disjoint
  frames," which was too strict for crop-as-extrinsic or same-time
  different-query supervision. The final contract says exact target
  observations must be disjoint from encoder input, while same source time is
  allowed for genuinely different queries whose target pixels were not already
  visible to the encoder.
- Updated `how_to_think_about_architecture.md` mistake #18 so it no longer
  repeats the old "subsumption theorem" language. The new rule is predictive
  quotient before mechanism, plus rate/minimality for representative
  selection.
- Updated `AGENTS.md`, `research_notes/README.md`,
  `research_notes/framing_the_problem/README.md`,
  `research_notes/meta_philosophy/README.md`, and
  `agent_notes/key_learnings.md` so future agents route to framing 3 plus the
  training contract rather than inventing `framing_4.md`.

## Current model

Framing 3 remains the default. It should be read as:

```text
W0 = E(O)
S_tau = G(W0, tau)
Ihat = R_fixed(S_tau, c)
L_v1 = L_pred(omitted observations on D_var support) + beta * Rate(W0)
```

The learned object is a minimal predictive representative modulo gauge on
supported queries. Stochastic world semantics are acknowledged, but full
stochastic export remains EH-3 unless measured multimodality forces it.

## Verification

- Ran `rg` scans for stale phrases: `subsumption theorem`, `subsumes every`,
  `single photometric`, `one photometric`, `Decode(W, c`, `F1-F6`, and
  related variants.
- Ran `git diff --check`; no whitespace errors.

No code tests were run because the work was documentation-only.
