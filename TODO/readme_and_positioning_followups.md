# README And Positioning Follow-Ups

## Context

The README has moved to a two-stage shape: first train `video => world tokens`,
then train AR or diffusion models that predict world tokens for generation.
World tokens are scene-state tokens that decode to splats and remain consistent
under novel cameras.

User also defined the eval bar in-chat: *as good as the video that comes out of
the base video model.* Same-camera round-trip tests adapter quality;
novel-camera tests whether the tokens are actually world tokens. The bar
matches the method: frozen backbone caps adapter ceiling, while stage-2
generation should predict token trajectories rather than patch a broken
representation.

These follow-ups are the gaps a skeptical first-time reader hits.

## Docs Follow-Ups

1. Add a short "What 'working' means" paragraph to the README, probably under
   Phase I. Two sentences max, user's exact phrasing:
   - same-camera re-render should match what the base video model would emit
     for that clip
   - novel-camera re-render should look consistent with it (no floaters, no
     gaps, realistically consistent geometry)
   - must be landed in user voice, not LLM-polished. Confirm wording before
     committing prose.

2. Add one sentence connecting `Video <=> Video` and `Cheap Adapters` so the
   training loop is explicit: the backbone is frozen, the splat head plus
   renderer is the adapter, reconstruction loss through the renderer trains
   the adapter. Without this, the two sections read as independent claims.

3. Keep the stage split explicit in the README: stage 1 is `video => world
   tokens`; stage 2 is AR/diffusion over world tokens for video continuation,
   image=>video, and text=>video. Do not let generation become the proof that
   the base representation worked.

4. Reconcile `research_notes/README.md` with main `README.md`. The research
   notes still have an older "Key Beliefs & Postulates" list predating the
   modality-shift frame. Either subordinate research_notes to main README or
   rewrite to match. Do not let them disagree silently.

5. Add a LICENSE file. "Hollywood's Open Source World Model" with no license
   blocks anyone who reads it carefully. Default to Apache-2.0 or MIT unless
   there is a reason to differ.

## Research Follow-Ups

6. Flesh out the evaluation protocol as a short research_note:
   - same-camera round-trip loss: `base_video_model(clip) vs render(encode(clip), original_path)`
   - novel-camera perceptual check: LPIPS / SSIM against the backbone re-encoding a
     nearby real camera if available; qualitative otherwise
   - held-out-frame PSNR during pretrain (standard self-supervised signal)
   - which of these are training losses vs eval-only metrics

7. Write a short research_note on stage-2 world-token generation. Compare AR
   token prediction, token diffusion, and hybrid rolling diffusion for video
   continuation, image=>video, and text=>video. Keep the question separate from
   novel-camera consistency, which belongs to the base representation.

8. Pretraining pressure experiments (already listed under "World Token Base Model"
   todos in the README). Worth promoting at least one to a proper research
   plan:
   - same-video chunk mixing: encode chunk A, swap in chunk B's camera token,
     train against chunk B's GT
   - crop-as-extrinsic: define the crop shift as a camera extrinsic change,
     see if the model learns to respect it
   - both together plus half-clip preimage forcing

9. Multi-camera loss per step (from `key_learnings.md` bullet 67): GT-camera
   photometric loss plus perturbed-camera diffusion-as-loss in a single step.
   This is the base-model answer to camera leakage. It is compatible with later
   stage-2 generation, but should not be replaced by it.

## Demo / Artifact Follow-Ups

10. Produce a minimal public demo clip: embed, change camera path, re-render.
   Even a low-fidelity version on the 128px/4fps fast-mac-gsplat baseline is
   worth more for the README than another bullet list. Pitch reads
   differently with one 10-second clip attached.

11. Video<=>splats round-trip demo. Encode a short clip, decode splats, render
    back to video, show side-by-side with input. Tests the "modality shift
    codec" claim visually.

## Guardrails

- Voice is preserved in `soul_documents/voice.md`. Re-read before any README
  prose pass. Do not LLM-polish user phrasings into generic research-lab
  voice.
- Do not pitch generation as stage 1. Text=>video and image=>video belong to
  stage 2, after world tokens pass novel-camera consistency checks.
- Do not name external video-diffusion models (Sora, Veo, Wan, SVD) as
  structural hooks in README prose.
- No parallel-bullet slop. No colon-subtitle headers. No em-dash flourishes.
- Keep CORE_GOAL alignment: MVP is import video, train splat, edit camera,
  bake render. Research notes can go deeper; the README top half must stay
  close to that.
