# README Voice Cleanup

## Context

Claude rewrote `README.md` and added `soul_documents/voice.md`, but the pass
smoothed too much of Nicholas's actual phrasing and reintroduced a planned
text/image generative-world-model Phase III.

## Changes

- Reworked `README.md` around the real n-grams:
  - `Dynamic video => compressed splats`
  - `Video <=> splats`
  - `Video <=> Video`
  - "less about making new worlds and more about modality shifts"
  - "they are complementary, not opposites"
  - "There is not a generative phase planned"
  - "it only generates novel angles"
  - "train on the GT of images it didn't encode"
- Added the second use case: special effects on splat representations, where
  traditional physics pipelines and algorithmic effects provide control that
  video diffusion may not.
- Rewrote `soul_documents/voice.md` to preserve high-conviction chat n-grams
  instead of describing a generic terse startup voice.
- Added `research_notes/loose_thoughts/ideation_on_what_is_a_world_model.md`
  with the raw thought dump plus a separate formalization attempt.

## Verification

Docs-only pass. No runtime tests needed.
