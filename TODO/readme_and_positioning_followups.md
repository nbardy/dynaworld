# README And Positioning Follow-Ups

## Context

The README has stabilized around: modality shift, `Video <=> Video` as the
training contract, cheap adapters on a frozen video backbone, no text-to-world
Phase III, bounded generative capacity for novel angles only.

User also defined the eval bar in-chat: *as good as the video that comes out of
the base video model.* Same-camera round-trip tests adapter quality;
novel-camera tests generative capacity on top. The bar matches the method —
frozen backbone caps adapter ceiling.

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

3. Reconcile `research_notes/README.md` with main `README.md`. The research
   notes still have an older "Key Beliefs & Postulates" list predating the
   modality-shift frame. Either subordinate research_notes to main README or
   rewrite to match. Do not let them disagree silently.

4. Add a LICENSE file. "Hollywood's Open Source World Model" with no license
   blocks anyone who reads it carefully. Default to Apache-2.0 or MIT unless
   there is a reason to differ.

## Research Follow-Ups

5. Flesh out the evaluation protocol as a short research_note:
   - same-camera round-trip loss: `base_video_model(clip) vs render(encode(clip), original_path)`
   - novel-camera perceptual check: LPIPS / SSIM against the backbone re-encoding a
     nearby real camera if available; qualitative otherwise
   - held-out-frame PSNR during pretrain (standard self-supervised signal)
   - which of these are training losses vs eval-only metrics

6. Write a short research_note on the generative-capacity mechanism. The
   README commits to bounded hallucination (novel angles, hidden geometry,
   background extension) but is hand-wavy about how. The Video Diffusion
   Bootstrap todos hint at score distillation and single-step zero-SNR
   features; pick one, describe how gradients flow from diffusion features
   back into the renderer, and what the loss is at novel camera positions
   where no GT exists.

7. Pretraining pressure experiments (already listed under "Model Architecture"
   todos in the README). Worth promoting at least one to a proper research
   plan:
   - same-video chunk mixing: encode chunk A, swap in chunk B's camera token,
     train against chunk B's GT
   - crop-as-extrinsic: define the crop shift as a camera extrinsic change,
     see if the model learns to respect it
   - both together plus half-clip preimage forcing

8. Multi-camera loss per step (from `key_learnings.md` bullet 67): GT-camera
   photometric loss plus perturbed-camera diffusion-as-loss in a single step.
   This is the one-stage answer to the two-stage AR novel-view trap. Worth a
   proper design note before anyone tries it.

## Demo / Artifact Follow-Ups

9. Produce a minimal public demo clip: embed, change camera path, re-render.
   Even a low-fidelity version on the 128px/4fps fast-mac-gsplat baseline is
   worth more for the README than another bullet list. Pitch reads
   differently with one 10-second clip attached.

10. Video<=>splats round-trip demo. Encode a short clip, decode splats, render
    back to video, show side-by-side with input. Tests the "modality shift
    codec" claim visually.

## Guardrails

- Voice is preserved in `soul_documents/voice.md`. Re-read before any README
  prose pass. Do not LLM-polish user phrasings into generic research-lab
  voice.
- No new Phase III. Novel angles only. Background extension only. No
  text-to-world.
- Do not name external video-diffusion models (Sora, Veo, Wan, SVD) as
  structural hooks in README prose.
- No parallel-bullet slop. No colon-subtitle headers. No em-dash flourishes.
- Keep CORE_GOAL alignment: MVP is import video, train splat, edit camera,
  bake render. Research notes can go deeper; the README top half must stay
  close to that.
