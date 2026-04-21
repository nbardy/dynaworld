# README Positioning Wrap

## Context

Continuation of the README voice/positioning work started in
`2026-04-20_14-35-38_readme_voice_cleanup.md`. This thread iterated the pitch
from "generic open-source world model" through several frames and landed on a
tighter claim set with an explicit "what working looks like" bar.

## Positioning Moves That Landed

- Kept the `Dynamic video => compressed splats` one-liner and the `Video <=>
  splats` / `Video <=> Video` arrows. These are load-bearing n-grams in
  `soul_documents/voice.md`; do not smooth them.
- Kept "Dynaworld is less about making new worlds and more about modality
  shifts" as the primary frame. Exploratory vs. generative is a taxonomy move
  underneath, not the headline.
- Added a scaling claim to the `Video <=> Video` section: *`Video <=> Video`
  is the only training data for world models that scales. Everything else
  requires expensive labeling, and labels don't scale.* This is the
  justification for why the training contract is self-supervised on video,
  not why splats specifically sit in the middle.
- Added a new `## Cheap Adapters` section: *Modalities don't require
  pretraining. The implicit latents in video models are the key. Decoding them
  to splats is cheap adapter training. A lightweight splat head on a frozen
  video backbone. Not a new foundation model.* This is the "why this is
  tractable" claim — it is not redundant with Video<=>Video.
- Core Beliefs bullets 3 and 6 were updated to fold both claims into the belief
  list so the README reads coherently top-to-bottom.

## Positioning Moves That Were Rejected

- Phase III text-to-world generation. User killed this explicitly. Do not
  reintroduce it. "DynaWorld generates novel angles, not starting worlds."
- Naming external video-diffusion models (Sora / Veo / Wan / SVD) as structural
  hooks. Voice slop.
- YC-landing-page phrasings. Parallel-bullet slop. Colon-subtitle headers. Em-
  dash flourishes. All flagged in `soul_documents/voice.md`.
- `Phase II = SFX`. Phase II is interaction. SFX is a use case under Phase I
  splat representations.

## What "Working" Means (new, not yet in README)

User stated the eval bar in-chat:

- embed a video, change the camera path, re-render, get "perfect" novel-view
  synthesis with no floaters, no gaps, realistically consistent geometry.
- "perfect" was then clarified: *as good as the video that comes out of the
  base video model.*

Why this is structurally good, not just aspirational: if the video backbone is
frozen and the splat head is an adapter, then the ceiling of the adapter is
the backbone's own reconstruction quality. The eval bar matches the method —
it is not a wish layered on top.

Same-camera re-render tests adapter quality. Novel-camera re-render tests
generative capacity on top. Both are falsifiable against the backbone.

This paragraph was discussed but **not yet written into the README**. Pending
user confirmation on exact wording before committing prose.

## Voice Guardrails Worth Repeating

- `voice.md` now contains a "Slop Detected In The Last Pass" section. Treat
  those as durable bans, not one-time corrections.
- Before shipping README prose, scan for: neat market maps, fake Phase IIIs,
  external name drops as structure, three-sentence parallel bullets,
  stacked-adjective sentences, colon-subtitle headers.
- Verbatim n-grams from chat should be preferred to smoothed variants, even
  when the smoothed version reads "better." Voice preservation beats polish.

## Open Tensions Flagged But Not Fixed

These are real and worth addressing in a later pass. Captured in
`TODO/readme_and_positioning_followups.md`:

1. `research_notes/README.md` vs main `README.md` drift. The research notes
   still have an older "Key Beliefs" list predating the modality-shift frame.
2. The training-loop connection between "Video<=>Video" and "cheap adapters"
   is implicit. Readers have to infer that reconstruction loss on video closes
   the loop through a frozen backbone + trainable adapter. One explicit
   sentence would harden the pitch.
3. No demo / no eval metric anywhere in the README. The pitch sells a
   capability; a skeptical reader (Karpathy-mode) asks "show me the clip"
   immediately.
4. "Generative Capacity" section is honest about the need (novel angles
   require hallucinating unseen content) but hand-wavy about mechanism. The
   Video Diffusion Bootstrap todos are the real mechanism and they are buried
   under a todo checklist.
5. No LICENSE file. "Hollywood's Open Source World Model" with no license is a
   contradiction.

## Files Touched This Thread

- `README.md`: added scaling claim in `Video <=> Video`, added `## Cheap
  Adapters` section, updated Core Beliefs bullets 3 and 6. User also rewrote
  the "Generative Capacity" section to remove Phase III and carve out bounded
  hallucination.
- `soul_documents/voice.md`: user heavily rewrote to add anti-slop rules,
  "Keep These Moves", "Do Not Normalize These Away", and verbatim chat
  n-grams including the new generative-capacity phrasings.
- No code changes this thread.

## Verification

Docs-only thread. No runtime tests needed. No trainer changes.
