# World/splat token philosophy doc

Session on 2026-04-24. User asked for a large philosophy document clarifying
the difference between world tokens, splat tokens, and text/image/video tokens,
with special focus on why a two-stage splat tokenizer is not automatically
valid when there are no ground-truth splats to tokenize.

## What changed

- Added
  `research_notes/meta_philosophy/world_splat_tokens_vs_observed_modality_tokens.md`.
- Updated `research_notes/meta_philosophy/README.md` so the new doc sits in the
  strategic reading order before the architecture mistakes checklist.
- Updated `research_notes/README.md` so the new token philosophy doc is visible
  from the top-level research navigation.

## Core content

- Text/image/video tokens are tokens of observed data; world/splat tokens are
  latent predictive assets inferred through render supervision.
- A same-view splat autoencoder/tokenizer can freeze source-view billboard
  degeneracy, then a predictor can faithfully learn that bad target.
- The default recommendation remains single-stage held-out predictive training:
  `W = E(O)`, `S_tau = G(W, tau_q)`,
  `image = R_fixed(S_tau, camera_q)`, with `H not_subset O` and rate/minimality.
- Two-stage training is legitimate only when the tokenizer/teacher is itself
  trained under the non-degenerate world contract and judged by render behavior,
  not raw token equality.
- Recommended representation split: one exported world asset `W`, with a typed
  time-conditioned splat readout at the renderer boundary.

## Verification

- Ran `git diff --check`; no whitespace errors.
- Counted the new doc at 1292 lines and verified the section outline with `rg`.

No code tests were run because this was documentation-only.
