# World Token README And Plan Update

User reframed the broad goal:

- first: train a `video => world token` base model;
- world tokens are only world tokens if they decode to splats that stay
  consistent across novel camera angles;
- follow-up: train AR or diffusion models over those world tokens for video
  continuation, image=>video, and text=>video.

This updates the older README guardrail that blocked text-conditioned
world/video generation. The new guardrail is sequencing, not prohibition:
generation belongs after the base representation works, and it should predict
world tokens rather than patch camera leakage in weak tokens.

Files touched:

- `README.md`: top-level thesis, training contract, phases, progress checklist,
  and future-direction bullets now say `video => world tokens => splats` first
  and world-token generation second.
- `research_notes/README.md`: core hypothesis and next steps now match the
  world-token base-model framing.
- `research_notes/potential_directions_index.md`: routing map now has a Now
  direction for the video-to-world-token base model and a speculative follow-up
  direction for world-token AR/diffusion generation.
- `TODO/readme_and_positioning_followups.md`: old generation-prohibition
  guardrail changed into "do not pitch generation as stage 1."

Reasoning note:

The existing `key_learnings.md` warning about two-stage AR novel-view repair is
still valid. Stage-2 token generation is acceptable only if stage 1 tokens
already satisfy source-camera and novel-camera consistency. If they do not, the
stage-2 model becomes a standalone video generator with extra baggage.
